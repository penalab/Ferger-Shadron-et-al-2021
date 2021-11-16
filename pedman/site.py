"""Site class for easy access and caching
"""

import os
import hashlib
from collections import OrderedDict
import numpy as np
import pandas as pd
from ._internals import LazyLoader
from . import meta
from . import pyxdphys as pyxd
from . import pypl2 as pypl2
if not hasattr(pypl2, 'pl2_spikeinfo'):
    # Make sure our extensions for pypl2 are loaded
    from . import extend_pypl2


def ordereddict_insert_after(od, after, newkey, value):
    od[newkey] = value
    od.move_to_end(newkey, last=False)
    keys = list(od.keys())
    for k in keys[keys.index(after):0:-1]:
        od.move_to_end(k, last=False)


def ordereddict_insert_before(od, before, newkey, value):
    od[newkey] = value
    od.move_to_end(newkey, last=False)
    keys = list(od.keys())
    for k in keys[keys.index(before)-1:0:-1]:
        od.move_to_end(k, last=False)


class Site():
    """Manage data and metadata for one recording site
    
    One recording site may also refer to parallel recordings with multiple
    electrodes - actually that is the initial intention.
    
    Arguments
    ---------
    metafile : str
        path to the site meta file (YAML file)
    basedir : str
    rec_selection : None (default) or 'all'
    unit_selection : None (default) or 'all'
    
    """
    
    def __init__(self, metafile, basedir = None, rec_selection = None, unit_selection = None):
        """Initialize a Site object with the absolute path to the meta file"""
        self.metafile = os.path.abspath(metafile)
        if basedir is not None:
            self.basedir = os.path.abspath(basedir)
        self.unit_selection = unit_selection
        self.rec_selection = rec_selection
        self.meta = meta.load(self.metafile)
        self.recs = RecordingContainer(self)
        self.units = UnitContainer(self)
        
    def id(self):
        return f"{self.meta['date'].strftime('%Y%m%d')}-{self.meta['site']:02d}"

    def __repr__(self):
        return f"Site({self.metafile!r})"

    def __str__(self):
        return f"Site[{self.id()}]"

    def unit_keys(self):
        return list(self.units.keys())
    
    def isunsorted(self):
        return all([u.meta['grade'] == 'unsorted' for u in self.units.values()])

    def hasunsorted(self):
        return any([u.meta['grade'] == 'unsorted' for u in self.units.values()])

    def get_dir(self, which=None):
        """Get absolute directory paths for this Site.
        
        which -- None     ... directory of meta file
                 'all'    ... common data directory
                 'plexon' ... directory of plexon files
                 'xdphys' ... directory of xdphys files
        """
        # Remark: os.path.join will automatically use given absolute path in
        # later arguments, such that os.path.join(somepath, someabspath) returns
        # someabspath without further need to check this.
        if which is None:
            return os.path.abspath(os.path.dirname(self.metafile))
        elif which == 'base':
            if hasattr(self, 'basedir') and self.basedir is not None:
                return self.basedir
            else:
                self.basedir = self.get_dir(None)
                return self.basedir
        elif which == 'all':
            return os.path.abspath(os.path.join(
                self.get_dir('base'),
                self.meta['directories'].get('all', "") or ""
            ))
        elif which in ['xdphys', 'plexon']:
            return os.path.abspath(os.path.join(
                self.get_dir('all'),
                self.meta['directories'].get(which, "") or ""
            ))
        else:
            raise ValueError(f"Cannot find '{which}' directory for {self!r}")

    def paradigms(self):
        return OrderedDict((rk, r.paradigm) for rk, r in self.recs.items())

    def has_xdphys(self, fileindex):
        return self.recs[fileindex].has_xdphys()
        
    def data_files(self):
        file_list = []
        for r in self.recs[:]:
            file_list.extend(r.data_files())
        return file_list
    
    def load_pl2_info(self, fileindex):
        """"""
        return self.recs[fileindex].pl2_info

    def load_pl2_events(self, fileindex):
        """"""
        return self.recs[fileindex].pl2_events

    def load_pl2_spikes(self, fileindex, channel):
        """"""
        return self.recs[fileindex].channels[channel].pl2_spikes

    def load_pl2_spikeinfo(self, fileindex, channel):
        """"""
        return self.recs[fileindex].channels[channel].pl2_spikeinfo

    def channels(self, fileindex = 0):
        return self.recs[fileindex].channels
        
    def stimtime(self, fileindex):
        return self.recs[fileindex].stimtime()
        
    def trialparams(self, fileindex):
        return self.recs[fileindex].trialparams()
        
    def trialtimes(self, fileindex):
        return self.recs[fileindex].trialtimes()

    def responses(self, fileindex, spikes_per_second=False):
        """Return responses in a Pandas DataFrame"""
        return self.recs[fileindex].responses(spikes_per_second = spikes_per_second)
        
    def get_calfile(self):
        if not hasattr(self, 'calfile'):
            self.calfile = pyxd.XDcal(
                os.path.join(self.get_dir('xdphys'), self.meta['calibration']['xdcalfile'])
            )
        return self.calfile
    
    ####
    ## Hash Functions
    ## see: https://gitlab.com/penalab/data_analysis/-/wikis/data/Hashes
    ####
    
    def update_hashes(self, save_metafile = False):
        regenerate_counts = {'plexon': 0, 'xdphys': 0}
        for r in self.recs[:]:
            # plexon hashes:
            plexon_hash = r.get_plexon_hash()
            if 'plexon_hash' not in r._meta or plexon_hash != r._meta['plexon_hash'] or 'plexon_file_hash' not in r._meta:
                # Plexon hashes missing OR quicker plexon_hash doesn't match, regenerate:
                ordereddict_insert_after(r._meta, 'plexon', 'plexon_hash', plexon_hash)
                ordereddict_insert_after(r._meta, 'plexon_hash', 'plexon_file_hash', r.get_plexon_file_hash())
                regenerate_counts['plexon'] += 1
            # xdphys hashes:
            if r.has_xdphys():
                xdphys_hash = r.get_xdphys_hash()
                if 'xdphys_hash' not in r._meta or xdphys_hash != r._meta['xdphys_hash'] or 'xdphys_file_hash' not in r._meta:
                    # xdphys hashes missing OR quicker xdphys_hash doesn't match, regenerate:
                    ordereddict_insert_after(r._meta, 'plexon_file_hash', 'xdphys_hash', xdphys_hash)
                    ordereddict_insert_after(r._meta, 'xdphys_hash', 'xdphys_file_hash', r.get_xdphys_file_hash())
                    regenerate_counts['xdphys'] += 1
        if save_metafile and sum(regenerate_counts.values()) > 0:
            pedman.meta.dump(self.meta, self.metafile)
        return regenerate_counts


class Unit():
    """
    """

    def __init__(self, site, unitname):
        self.site = site
        self.name = unitname
        self.meta = site.meta['units'][self.name]
        self._cache = {}

    def __repr__(self):
        return f"{self.site!r}.units[{self.name!r}]"

    def __str__(self):
        return f"Unit[{self.site.id()}.{self.name}]"

    def channel(self):
        return self.meta['channel']
        
    def channel_name(self, fileindex):
        return self.site.recs[fileindex].channels[self.channel()].name
        
    def sortcode(self, fileindex = None):
        if fileindex is not None and ('sortcode_exceptions' in self.meta and
                self.meta['sortcode_exceptions'] is not None and
                fileindex in self.meta['sortcode_exceptions']):
            return self.meta['sortcode_exceptions'][fileindex]
        else:
            return self.meta['sortcode']

    def has_xdphys(self, fileindex):
        return self.site.recs[fileindex].has_xdphys()

    def spiketimes(self, fileindex):
        pl2_spikes = self.site.recs[fileindex].channels[self.channel()].pl2_spikes
        return np.asarray(pl2_spikes.timestamps)[np.asarray(pl2_spikes.units) == self.sortcode(fileindex)]

    def spiketrains(self, fileindex, prestim = 0, poststim = 0, dur = None):
        rec = self.site.recs[fileindex]
        # Calculate timings:
        stimstart, stimend = rec.stimtime()
        if dur is not None:
            stimend = stimstart + dur
        evt_start = stimstart - prestim # time _added_ to event time
        evt_end = stimend + poststim
        # Caching (cache key based on timings instead of parameters to capture more cases):
        cache_key = ('spiketrains', rec.index, evt_start, evt_end)
        if cache_key in self._cache:
            return self._cache[cache_key]
        events = rec.trialtimes()
        spikes = iter(self.spiketimes(fileindex))
        spiketrains = [[] for _ in range(events.size)]
        next_spike = -1
        for kevt, evt in enumerate(events):
            win_start = evt + evt_start
            win_end = evt + evt_end
            try:
                while next_spike <= win_start:
                    next_spike = next(spikes)
                while next_spike <= win_end:
                    spiketrains[kevt].append(next_spike - (evt + stimstart))
                    next_spike = next(spikes)
            except StopIteration:
                # No more spikes
                break
        self._cache[cache_key] = spiketrains
        return spiketrains

    def waveforms(self, fileindex):
        pl2_spikes = self.site.recs[fileindex].channels[self.channel()].pl2_spikes
        return np.array(pl2_spikes.waveforms)[np.array(pl2_spikes.units) == self.sortcode(fileindex)]

    def responses(self, fileindex, spikes_per_second=False):
        rec = self.site.recs[fileindex]
        cache_key = ('responses', rec.index, spikes_per_second)
        if cache_key in self._cache:
            return self._cache[cache_key]
        stimstart, stimend = rec.stimtime()
        df = rec.trialparams()
        index_columns = list(df.columns)
        events = rec.trialtimes()
        spikes = iter(self.spiketimes(fileindex))
        r = np.zeros(events.size)
        next_spike = -1
        for kevt, evt in enumerate(events):
            win_start = evt + stimstart
            win_end = evt + stimend
            try:
                while next_spike <= win_start:
                    next_spike = next(spikes)
                while next_spike <= win_end:
                    r[kevt] += 1
                    next_spike = next(spikes)
            except StopIteration:
                # No more spikes
                break
        df[self.name] = np.array(r) / (stimend - stimstart if spikes_per_second else 1.0)
        df = df.set_index(index_columns, append=True)
        self._cache[cache_key] = df
        return df

    def spontrates(self, fileindex, dur=None, prestart=0, spikes_per_second=False):
        """
        dur ... length of the spontaneous window, defaults to stimulus duration
        prestart ... the time between END of spontaneous window to start of stimulus
        """
        rec = self.site.recs[fileindex]
        cache_key = ('spontrates', rec.index, dur, prestart, spikes_per_second)
        if cache_key in self._cache:
            return self._cache[cache_key]
        stimstart, stimend = rec.stimtime()
        if dur is None:
            dur = stimend - stimstart
        df = rec.trialparams()
        index_columns = list(df.columns)
        events = rec.trialtimes()
        spikes = iter(self.spiketimes(fileindex))
        r = np.zeros(events.size)
        next_spike = -1
        for kevt, evt in enumerate(events):
            win_start = evt + stimstart - prestart - dur
            win_end = evt + stimstart - prestart
            try:
                while next_spike <= win_start:
                    next_spike = next(spikes)
                while next_spike <= win_end:
                    r[kevt] += 1
                    next_spike = next(spikes)
            except StopIteration:
                # No more spikes
                break
        df[self.name] = np.array(r) / (stimend - stimstart if spikes_per_second else 1.0)
        df = df.set_index(index_columns, append=True)
        self._cache[cache_key] = df
        return df
    
    def get_best_itd(self):
        """Return the unit's best ITD
        """
        if 'best_itd' in self._cache:
            return self._cache['best_itd']
        # Responses from first itd_bc recording:
        responses = self.responses(('itd_bc', 0))
        x, y = responses.query('bc == 100').groupby('itd').mean().reset_index().T.values
        # Code adapted from tools.curve_analysis.peak_position.half_height_position
        max_I = np.argmax(y)
        half_max_Y = np.mean(y) + np.std(y)
        l = np.where(y[:max_I] <= half_max_Y)[0][-1]
        r = np.where(y[max_I+1:] <= half_max_Y)[0][0] + max_I
        H = lambda n: (x[n+1]-x[n]) * ((half_max_Y-y[n]) / (y[n+1]-y[n])) + x[n]
        best_itd = np.mean((H(l), H(r)))
        self._cache['best_itd'] = best_itd
        return best_itd
    
    def get_max_response(self, bc = 100):
        """Return the unit's best ITD
        """
        cache_key = ('max_response', bc)
        if cache_key in self._cache:
            return self._cache[cache_key]
        # Responses from first itd_bc recording:
        responses = self.responses(('itd_bc', 0))
        dg = responses.query('bc == @bc').groupby(responses.index.names[1:]).mean()
        max_response = dg.max().iat[0]
        self._cache[cache_key] = max_response
        return max_response
    
    def get_best_freq(self, force_recalculation = False):
        """Return the unit's best Freq
        """
        if not force_recalculation and 'best_freq' in self._cache:
            return self._cache['best_freq']
        # Responses from first bf recording:
        responses = self.responses(('bf', 0))
        itd_with_max_responses = responses.groupby(['itd', 'stim']).mean().reset_index('itd').groupby('itd').max().idxmax().values[0]
        x, y = responses.query('itd == @itd_with_max_responses').groupby('stim').mean().reset_index().T.values
        # Code adapted from tools.curve_analysis.peak_position.half_height_position
        max_I = np.argmax(y)
        if True:
            # Half-max method:
            half_max_Y = y[max_I] / 2
        else:
            # mean + std of spont rate:
            spont = self.spontrates(('bf',0))[self.name].values
            half_max_Y = spont.mean() + 2 * spont.std()
        l = np.where(y[:max_I] <= half_max_Y)[0][-1]
        r = np.where(y[max_I+1:] <= half_max_Y)[0][0] + max_I
        H = lambda n: (x[n+1]-x[n]) * ((half_max_Y-y[n]) / (y[n+1]-y[n])) + x[n]
        best_freq = np.mean((H(l), H(r)))
        self._cache['best_freq'] = best_freq
        self._cache['freq_lo'] = H(l)
        self._cache['freq_hi'] = H(r)
        return best_freq
    
    def get_min_freq(self):
        if not ('freq_lo' in self._cache):
            self.get_best_freq(True)
        return self._cache['freq_lo']
    
    def get_max_freq(self):
        if not ('freq_hi' in self._cache):
            self.get_best_freq(True)
        return self._cache['freq_hi']
    
    def get_freq_range(self):
        if not ('freq_lo' in self._cache and 'freq_hi' in self._cache):
            self.get_best_freq(True)
        return self._cache['freq_lo'], self._cache['freq_hi']


class UnitContainer():
    """
    """
    
    def __init__(self, site):
        self.site = site
        self.reload_units()
    
    def reload_units(self):
        units_meta = self.site.meta['units']
        # See if all units are unsorted (sortcode == 0):
        allunsorted = all([um.get('sortcode', None) == 0 for um in units_meta.values()])
        self._units = OrderedDict()
        for ku in units_meta.keys():
            if allunsorted:
                 # Don't skip units if all are unsorted
                 # TODO: This behavior might change in the future
                pass
            elif getattr(self.site, 'unit_selection', None) == 'all':
                 # Don't skip units if site.unit_selection == 'all'
                pass
            else:
                # Select units based on grade and/or use values:
                if 'use' in units_meta[ku]:
                    # Overwrites selection based on grade:
                    if not units_meta[ku]['use']:
                        continue
                elif units_meta[ku].get('grade', None) in ['junk', 'unsorted']:
                    continue
            # Finally, add unit object:
            self._units[ku] = Unit(self.site, ku)
    
    def __getitem__(self, key):
        return self._units.__getitem__(key)
    
    def __contains__(self, key):
        try:
            return bool(self[key])
        except KeyError:
            return False
    
    def __iter__(self):
        return iter(self._units)
    
    def __len__(self):
        return len(self._units)
    
    def items(self):
        return self._units.items()
    
    def values(self):
        return self._units.values()
    
    def keys(self):
        return self._units.keys()


class Recording(metaclass=LazyLoader):


    def __init__(self, site, fileindex):
        self.site = site
        self.index = fileindex
        self._meta = site.meta['files'][self.index]
        for k in self._meta:
            try:
                setattr(self, k, self._meta[k])
            except:
                pass

    def __repr__(self):
        return f"{self.site!r}.recs[{self.index!r}]"
        
    def __str__(self):
        return f"{self.site}.recs[{self.index}]"
    
    def has_xdphys(self):
        return hasattr(self, 'xdphys') and self.xdphys is not None
    
    def has_plexon(self):
        return hasattr(self, 'plexon') and self.plexon is not None
    
    def xdphys_path(self):
        return os.path.join(self.site.get_dir('xdphys'), self.xdphys)
    
    def xdphys_paths(self):
        """Return a list of all xdphys paths, can be empty."""
        if self.has_xdphys():
            return [self.xdphys_path()]
        else:
            return []
    
    def plexon_path(self):
        return os.path.join(self.site.get_dir('plexon'), self.plexon)
        
    def data_files(self):
        file_list = []
        if self.has_plexon():
            file_list.append(self.plexon_path())
        if self.has_xdphys():
            file_list.extend(self.xdphys_paths())
        return file_list
    
    def load_xfile(self):
        if self.has_xdphys():
            self.xfile = pyxd.XDdata(self.xdphys_path())
        else:
            raise AttributeError(f"{self} has no xdphys file.")
    _lazy['xfile'] = "load_xfile"
    
    def matches_variant(self, variant = None):
        if variant is None:
            # variant None is always a match:
            return True
        if not hasattr(self, 'variant'):
            # Recording has no variant, thus cannot match:
            return False
        if isinstance(variant, str) and isinstance(self.variant, str):
            # str-variant
            return variant == self.variant
        if isinstance(variant, dict) and isinstance(self.variant, dict):
            return all(variant[kv] == self.variant.get(kv, None)
                for kv in variant.keys()
            )
        return False

    def stimtime(self):
        dur = self.xfile.params['dur'] * 1e-3
        delay = self.xfile.params['delay'] * 1e-3
        return (delay, delay + dur)

    def trialparams(self):
        return pd.DataFrame(self.xfile.trials).set_index("index")

    def load_pl2_info(self):
        """"""
        self.pl2_info = pypl2.pl2_info(self.plexon_path())
    _lazy['pl2_info'] = "load_pl2_info"

    def load_pl2_events(self):
        """"""
        self.pl2_events = pypl2.pl2_events(
            self.plexon_path(),
            self.pl2_info.events[0].name.decode('ascii')
        )
    _lazy['pl2_events'] = "load_pl2_events"

    def load_channels(self):
        self.channels = OrderedDict(
            (si.channel, PlexonChannel(self, si))
            for si in self.pl2_info.spikes
        )
        self.channels_by_name = OrderedDict(
            (c.name, c)
            for c in self.channels.values()
        )
    _lazy['channels'] = "load_channels"
    _lazy['channels_by_name'] = "load_channels"

    def trialtimes(self):
        times = np.array(self.pl2_events.timestamps)
        if hasattr(self, 'plexon_event_range'):
            times = times[self.plexon_event_range[0]:self.plexon_event_range[1]]
        return times

    def responses(self, spikes_per_second=False):
        """Return responses in a Pandas DataFrame"""
        df = self.trialparams()
        df.set_index(list(df.columns), append = True, inplace = True)
        for ku, u in self.site.units.items():
            df = df.join(u.responses(self.index, spikes_per_second=spikes_per_second))
        return df

    def spontrates(self, dur=None, prestart=0, spikes_per_second=False):
        """Return spontrates in a Pandas DataFrame"""
        df = self.trialparams()
        df.set_index(list(df.columns), append = True, inplace = True)
        for ku, u in self.site.units.items():
            df = df.join(u.spontrates(self.index, dur=dur, prestart=prestart, spikes_per_second=spikes_per_second))
        return df
        
    def load_calfile(self):
        self.calfile = self.site.get_calfile()
    _lazy['calfile'] = 'load_calfile'
    
    def calc_kstim(self):
        """Calculate the index of the actual stimulus for Recording `self`."""
        self.stimstart_ind = int(np.floor(self.xfile.params['delay'] * 1e-3 * self.xfile.params['dafc']))
        self.stimstop_ind = self.stimstart_ind + int(np.floor(self.xfile.params['dur'] * 1e-3 * self.xfile.params['dafc'])) + 1
    _lazy['stimstart_ind'] = 'calc_kstim'
    _lazy['stimstop_ind'] = 'calc_kstim'
    
    def load_stimulus_decalibrated(self):
        """"""
        # Get calibrated (raw) stimuli:
        stim_L = self.xfile.stimulus[:, self.stimstart_ind:self.stimstop_ind, 0]
        stim_R = self.xfile.stimulus[:, self.stimstart_ind:self.stimstop_ind, 1]
        # Calculate FFT of calibrated stimuli:
        stim_L_fft = np.fft.rfft(stim_L)
        stim_R_fft = np.fft.rfft(stim_R)
        # Get Calibration data, interpolated for fft frequencies:
        freqs = np.fft.rfftfreq(stim_L.shape[1], 1/self.xfile.params['dafc'])
        caldata_smooth = self.calfile.interpolate_caldata(freqs)
        # Normalize magnitudes and convert from dB to amplitude
        mag_min_LR = np.nanmin(caldata_smooth[['left_mag', 'right_mag']].values)
        scale_L = 10 ** ((mag_min_LR - caldata_smooth['left_mag'].values) / -20)
        scale_R = 10 ** ((mag_min_LR - caldata_smooth['right_mag'].values) / -20)
        # Calculate phase shifts (complex representation)
        shift_L = np.exp( 1j * caldata_smooth['left_phase'].values )
        shift_R = np.exp( 1j * caldata_smooth['right_phase'].values )
        # Decalibrate for all frequencies we can, leave the remaing range untouched
        isfinite = np.isfinite(scale_L)
        stim_L_fft[:, isfinite] = (stim_L_fft * scale_L * shift_L)[:, isfinite]
        stim_R_fft[:, isfinite] = (stim_R_fft * scale_R * shift_R)[:, isfinite]
        # Get decalibrated stimuli by inverse FFT:
        stim_L_decal = np.fft.irfft(stim_L_fft)
        stim_R_decal = np.fft.irfft(stim_R_fft)
        self.stimulus = stim_L_decal, stim_R_decal
    _lazy['stimulus'] = 'load_stimulus_decalibrated'
    
    ####
    ## Hash Functions
    ## see: https://gitlab.com/penalab/data_analysis/-/wikis/data/Hashes
    ####
    
    @staticmethod
    def hash_file(filepath, update_hash = None, blocksize = 1024 * 1024 * 4):
        if update_hash is None:
            h = hashlib.sha1()
        else:
            h = update_hash
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(blocksize)
                h.update(chunk)
                if len(chunk) < blocksize:
                    break
        if update_hash is None:
            return h.hexdigest()
    
    def get_xdphys_hash(self):
        if not self.has_xdphys():
            return None
        h = hashlib.sha1()
        if hasattr(self, 'xfile'):
            with open(self.xfile.path, 'rb') as f:
                h.update(f.read(self.xfile._seek_rasterdata))
        else:
            for xfile in self.xfiles:
                with open(xfile.path, 'rb') as f:
                    h.update(f.read(xfile._seek_rasterdata))
        return h.hexdigest()
    
    def get_xdphys_file_hash(self):
        if not self.has_xdphys():
            return None
        if hasattr(self, 'xfile'):
            return self.hash_file(self.xfile.path)
        else:
            return [self.hash_file(p) for p in self.xdphys_paths()]
    
    def get_plexon_hash(self, force_reload = False):
        if not self.has_plexon():
            return None
        if force_reload:
            self.load_pl2_info()
        return hashlib.sha1(
            np.array([spks.units for spks in self.pl2_info.spikes]).tobytes()
        ).hexdigest()
    
    def get_plexon_file_hash(self, force_reload = False):
        if not self.has_plexon():
            return None
        return self.hash_file(self.plexon_path())


class RecordingMultiXDfiles(Recording):
    """A specialized class to handle recordings where one Plexon file is paired
    with multiple xdphys files. In meta files, this looks like:
    
    9:
        paradigm: adapt
        xdphys:
        -   039.01.8.abi
        -   039.01.9.abi
        -   039.01.10.abi
        -   039.01.11.abi
        -   039.01.12.abi
        -   039.01.13.abi
        -   039.01.14.abi
        -   039.01.15.abi
        -   039.01.16.abi
        -   039.01.17.abi
        plexon: 2019-0117-039-01-adapt-bb.pl2
    
    """
    
    def xdphys_path(self, index = None):
        if index is None:
            raise ValueError(f"There are multiple xdphys path for {self}")
        return os.path.join(self.site.get_dir('xdphys'), self.xdphys[index])
    
    def xdphys_paths(self):
        """Return a list of all xdphys paths, can be empty."""
        # has_xdphys() must be true
        return [self.xdphys_path(kx) for kx in range(len(self.xdphys))]
    
    def load_xfile(self):
        if self.has_xdphys():
            self.xfiles = [pyxd.XDdata(self.xdphys_path(index = i))
                           for i in range(len(self.xdphys))]
            # self.xfile = self.xfiles[0]
    del _lazy['xfile']
    _lazy['xfiles'] = 'load_xfile'
    
    def stimtime(self):
        dur = self.xfiles[0].params['dur'] * 1e-3
        delay = self.xfiles[0].params['delay'] * 1e-3
        return (delay, delay + dur)
        
    def trialparams(self):
        all_trials = []
        for xfile in self.xfiles:
            all_trials.extend(xfile.trials)
        df = pd.DataFrame(all_trials).drop('index', axis=1)
        df.index.name = 'index'
        return df


class RecordingContainer():
    """Container class to provide dict-like access to the recordings of a Site.
    
    Access recordings (when `s` is a Site object)
    
    ... by existing recordingindex -> single Recording object:
    
        s.recs[2]
        
    ... by paradigm -> list of Recording objects:
    
        s.recs['itd']
        s.recs['bf']
        s.recs['itd_bc']
    
    ... by paradigm and numeric index -> single Recording object:
    
        s.recs['itd', 0] # the first
        s.recs['itd', 1] # the second
        s.recs['itd', -1] # the last
    
    ... by paradigm and variant -> list of Recording objects:
    
        s.recs['two_snd', {'itd': 100}]
        s.recs['myparadigm', 'string-variant']
    
    ... by paradigm, variant and numeric index -> single Recording object:
    
        s.recs['two_snd', {'itd': 100}, 0]
    
    ... by slice object -> list of Recording objects:
    
        s.recs[:]
        s.recs[:2]
        s.recs[-2:]
    
    ... by paradigm and slice -> list of Recording objects, or
    ... by paradigm, variant and slice -> list of Recording objects:
    
        s.recs['itd', :2]
        s.recs['itd', 1:]
        s.recs['two_snd', {'itd': 100}, :-1:2]
    
    To test if a specific recording exists, you can use the usual membership
    test syntax. It uses the above indexing before testing and returns True only
    if one recording is found or a list of at least one recording.
    
        # has an 'itd' recording:
        'itd' in s.recs
        # has a 'two_snd' recording with variant-itd 100:
        ('two_snd', {'itd': 100}) in s.recs
        # has a third 'itd' recording:
        ('itd', 2) in s.recs
    
    """
    
    @staticmethod
    def _get_recording(site, fi):
        if isinstance(site.meta['files'][fi]['xdphys'], list):
            return RecordingMultiXDfiles(site, fi)
        else:
            return Recording(site, fi)
    
    def __init__(self, site):
        self.site = site
        self.reload_recs()
    
    def reload_recs(self):
        files_meta = self.site.meta['files']
        self._recs = OrderedDict()
        for fi in files_meta.keys():
            if 'use' in files_meta[fi] and not files_meta[fi]['use'] \
                and getattr(self.site, 'rec_selection', None) != 'all':
                # Skip files when `use` is falsy, unless site.rec_selection is 'all'
                continue
            self._recs[fi] = self._get_recording(self.site, fi)
        # Create and populate lookup dict by paradigm:
        self._recs_by_paradigm = {p: [] for p in
            set(r.paradigm for r in self._recs.values())
        }
        for r in self._recs.values():
            self._recs_by_paradigm[r.paradigm].append(r)
    
    def __getitem__(self, key):
        """
        """
        try:
            if isinstance(key, slice):
                # key like [slice] -> list
                return list(self._recs.values())[key]
            elif isinstance(key, tuple):
                if len(key) == 2 and isinstance(key[1], (int, slice)):
                    # key like [paradigm, index] -> single Recording
                    # key like [paradigm, slice] -> list
                    return self[key[0]][key[1]]
                elif len(key) == 2 and isinstance(key[1], (str, dict)):
                    # key like [paradigm, variant] -> list
                    return [r for r in self._recs_by_paradigm[key[0]]
                        if r.matches_variant(key[1])
                    ]
                elif (len(key) == 3 and isinstance(key[1], (str, dict))
                        and isinstance(key[2], (int, slice)) ):
                    # key like [paradigm, variant, index] -> single Recording
                    # key like [paradigm, variant, slice] -> list
                    return self[key[0], key[1]][key[2]]
            elif key in self._recs:
                # key like [recordingindex] -> single Recording
                return self._recs[key]
            elif key in self._recs_by_paradigm:
                # key like [paradigm] -> list
                return self._recs_by_paradigm[key]
            # Key not found:
            raise KeyError
        except KeyError:
            raise KeyError(f"{self.site} has no recording with index {key}.")
        except IndexError:
            raise KeyError(f"{self.site} has no recording with index {key}.")
        except TypeError:
            raise KeyError(f"{self.site} has no recording with index {key}.")
    
    def __contains__(self, key):
        try:
            return bool(self[key])
        except KeyError:
            return False

    def __iter__(self):
        return iter(self._recs)
        
    def items(self):
        return self._recs.items()

    def values(self):
        return self._recs.values()

    def keys(self):
        return self._recs.keys()

class PlexonChannel(metaclass=LazyLoader):


    def __init__(self, recording, spikeinfo):
        self.rec = recording
        self.spikeinfo = spikeinfo
        self.number = self.spikeinfo.channel
        self.name = self.spikeinfo.name.decode('ascii')
        
    def __str__(self):
        return f"PlexonChannel({self.name})"
        
    def __repr__(self):
        return f"{self.rec!r}.channels[{self.number}])"

    def load_pl2_spikes(self):
        """"""
        self.pl2_spikes = pypl2.pl2_spikes(self.rec.plexon_path(), self.name)
    _lazy['pl2_spikes'] = "load_pl2_spikes"

    def load_pl2_spikeinfo(self):
        """"""
        self.pl2_spikeinfo = pypl2.pl2_spikeinfo(self.rec.plexon_path(), self.name)
    _lazy['pl2_spikeinfo'] = "load_pl2_spikeinfo"

    def load_continuous_wb(self):
        self.wb = ContinuousData(self, kind = 'WB')
    _lazy['wb'] = "load_continuous_wb"

    def load_continuous_fp(self):
        self.fp = ContinuousData(self, kind = 'FP')
    _lazy['fp'] = "load_continuous_fp"

    def load_continuous_spkc(self):
        self.spkc = ContinuousData(self, kind = 'SPKC')
    _lazy['spkc'] = "load_continuous_spkc"


class ContinuousData(metaclass=LazyLoader):
    """"""
    
    def __init__(self, channel, kind = 'WB'):
        self.channel = channel
        self.kind = kind
        
    def load_pl2_ad(self):
        """"""
        pl2_ad = pypl2.pl2_ad(self.channel.rec.plexon_path(),
            f'{self.kind}{self.channel.number:02d}'
        )
        self.fs = pl2_ad.adfrequency
        self.n = pl2_ad.n
        self.t0 = pl2_ad.timestamps[0]
        #self.fragmentcounts = self.pl2_ad.fragmentcounts
        self.ad = np.asarray(pl2_ad.ad)
    _lazy['fs'] = "load_pl2_ad"
    _lazy['n'] = "load_pl2_ad"
    _lazy['t0'] = "load_pl2_ad"
    _lazy['ad'] = "load_pl2_ad"
    
    def make_times(self):
        self.times = np.arange(self.n) / self.fs + self.t0
    _lazy['times'] = "make_times"
