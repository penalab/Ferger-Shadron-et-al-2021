"""Interface representation of xdphys files

Class `XDdata`
--------------

"""

import os.path
import re
import gzip
from types import MethodType
from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd

class XDbase():
    """Base class for different xdphys files.
    
    Used to avoid code duplication. Not intended to be used directly.
    
    Known subclasses:
    
    * XDdata (formerly XDfile)
    * XDcal
    """

    # For compatibility check, minimum supported xdphys version
    _XDPHYS_MIN_VER = '' # XDbase should be valid for all versions
    
    # Program name in fileinfo -- Must be set by subclasses
    _XDPHYS_PROG_NAME = ''

    # Mapping dictionary for lazy-evaluated attributes,
    # see __getattr__ for how it's used. Keys are attribute names, values are
    # function names. MUST be copied by subclasses before being altered!
    _lazy_dict = {}

    def __init__(self, filepath):
        """Create a new object, representing an xdphys file"""
        self.path = os.path.abspath(filepath)
        self._file = FileHandler(self.path)
        # Parse the first, special line. This is used to check if the file can
        # be read and is indeed an xdphys data file.
        if not self.fileinfo.ver >= self._XDPHYS_MIN_VER:
            raise RuntimeError(
                f"The class {self.__class__} does not support xdphys versions "
                f"below '{self._XDPHYS_MIN_VER}', but file {self.path} has "
                f"version '{self.fileinfo.ver}'. "
                f"If available, try using a '_legacy' version instead."
            )

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}(r"{self.path!s}")'

    def __str__(self):
        return f'{self.__class__.__name__}("{self.path!s}")'

    def __getattr__(self, name):
        """Get lazy loaded attributes
        
        This function is called whenever an attribute is accessed that is not
        yet in the objects `__dict__`. The proper parsing function will be
        triggered and the (now available) attribute is returned.
        """
        if name in self._lazy_dict:
            # Find the function needed to make attribute `name` available,
            # and execute it:
            getattr(self, self._lazy_dict[name])()
            assert name in self.__dict__, ("Executing function"
                f" `{self.__class__.__name__}.{self._lazy_dict[name]}` "
                f"did not make `{name}` "
                "available as promised.")
            # Return the (now available) attribute:
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' has not attribute '{name}'")

    def __dir__(self):
        return super().__dir__() + list(self._lazy_dict.keys()) + ['stimulus', 'trace']

    def _convert_value(self, value):
        """
        """
        assert isinstance(value, str), "_convert_value can only convert str objects"
        # Strip leading and trailing white space:
        value = value.strip()
        if "," in value:
            # Parse a list of values, recursively
            try:
                return self._convert_list(value)
            except:
                # unsupported list specification, leave as is
                pass
        if ":" in value:
            # Parse a range of values, recursively
            try:
                return self._convert_range(value)
            except:
                # unsupported range specification, leave as is
                pass
        try:
            # convert to int if possible AND str representation does not change
            if str(int(value)) == value:
                value = int(value)
        except ValueError:
            # conversion to int failed, let's try float:
            try:
                value = float(value)
            except:
                # neither conversion to int nor to float were satisfying
                pass
        return value

    def _convert_range(self, range_spec):
        r_start, r_stop, r_step = (self._convert_value(e) for e in range_spec.split(':'))
        r_n = ((r_stop - r_start) / r_step)+1
        if r_n % 1 == 0:
            r_n = int(r_n)
        else:
            raise ValueError("Unsupported range specification '{}'".format(range_spec))
        return np.linspace(r_start, r_stop, r_n)

    def _convert_list(self, list_spec):
        templist = [self._convert_value(e) for e in list_spec.split(',')]
        return np.concatenate([[e] if np.isscalar(e) else e for e in templist]).astype(np.float)

    def _clean_key(self, key):
        """Cleanup xdphys param keys
        Removes whitespace and illegal characters. Also converts to lowercase.
        """
        # For debugging, test if key is really a str:
        assert isinstance(key, str), "XDbase._clean_key(): key must be a `str` obj"
        # strip white space and convert to lowercase
        key = key.strip().lower()
        # remove illegal leading characters
        key = re.sub('^[^a-z_]+', '', key)
        # replace  other illegal characters by underscores, thereby
        # merge successive characters into one underscore
        key = re.sub('[^0-9a-z_]+', '_', key)
        return key

    def _parse_infoline(self):
        """Read the first line of an xdphys file.
        
        Typically, the first line looks like this:
        
            ;; xdphys gen ver 2.8.1-1
        
        This line contains the recording `mod` (like 'gen', 'abi', 'itd', ...)
        and the xdphys version number (like '2.8.1-1')
        
        Sets attributes:
        
        - fileinfo
        - _seek_comments
        """
        with self._file as f:
            f.seek(0) # We know this must be the first line
            l = f.nextline()
            self._seek_comments = f.tell()
        m = re.match(';; {} ([^ ]+) ver (.*)'.format(self._XDPHYS_PROG_NAME), l)
        if not m:
            raise ValueError(f"Cannot read fileinfo from file {self.path!r}. Probably not an xdphys file.")
        # Use a namedtuple to make fileinfo accessible by dot-syntax:
        fileinfo_tuple = namedtuple('fileinfo', ('mod', 'ver'))
        self.fileinfo = fileinfo_tuple(mod = m.group(1),ver = m.group(2))
    # Attributes made available by _parse_infoline():
    _lazy_dict['fileinfo'] = "_parse_infoline"
    _lazy_dict['_seek_comments'] = "_parse_infoline"
    
    def _parse_comments(self):
        with self._file as f:
            # seek COMMENTS line
            f.seek(self._seek_comments)
            # Next line must be "COMMENTS":
            if not f.nextline() == "COMMENTS":
                raise ValueError(f"COMMENTS expected at position {self._seek_comments}")
            self.comments = []
            # read next line:
            l = f.nextline()
            while not l == "END_COMMENTS":
                # add comments line by line without leading semicolons:
                self.comments.append(l.lstrip(';'))
                # read next line:
                l  = f.nextline()
            self._seek_params = f.tell()
    # Attributes made available by _parse_comments():
    _lazy_dict['comments'] = "_parse_comments"
    _lazy_dict['_seek_params'] = "_parse_comments"

    def _read_params_line(self, l):
        """Read one line `l` of the PARAMS block and add to self.params.
        """
        # Ignore lines starting with a semicolon (silently):
        if l.startswith(';'):
            # this is a comment, skip!
            return
            #with self._file as f:
            #    self.params[f'PARAMS_COMMENT_{f.tell()-len(l)-1}'] = l
            #return
        m = re.match("(?:([^.=]+)\.)?([^.=]+)=(.*)", l)
        if m:
            groupkey, key, val = m.groups()
            # Resolve special case for "ana-decimate" to make it consistent:
            if key.startswith('ana'):
                groupkey = 'ana'
                key = key[4:] # 4 == len('ana-') == len('ana_')
            if groupkey is not None:
                groupkey = self._clean_key(groupkey)
            key = self._clean_key(key)
            # parse values as int, float, list, ...
            val = self._convert_value(val)
            if groupkey is None:
                # Simple param of form `key=value`
                assert (key not in self.params) or self.params[key] == val, f"Duplicate PARAM {key}"
                self.params[key] = val
            else:
                # Grouped param of form `groupkey.key=value`
                if not groupkey in self.params:
                    self.params[groupkey] = OrderedDict()
                assert (key not in self.params[groupkey]) or (self.params[groupkey][key] == val), f"Duplicate PARAM {groupkey}.{key}"
                self.params[groupkey][key] = val
        else:
            raise ValueError('non-matching PARAM line: "{}"'.format(l))

    def _parse_params(self):
        with self._file as f:
            # seek PARAMS line
            f.seek(self._seek_params)
            # Next line must be "PARAMS":
            if not f.nextline() == "PARAMS":
                raise ValueError(f"PARAMS expected at position {self._seek_params}")
            self.params = OrderedDict()
            # read next line:
            l = f.nextline()
            while not l == "END_PARAMS":
                # add params line by line:
                self._read_params_line(l)
                # read next line:
                l  = f.nextline()
            self._seek_rasterdata = f.tell()
    # Attributes made available by _parse_params():
    _lazy_dict['params'] = "_parse_params"
    _lazy_dict['_seek_rasterdata'] = "_parse_params"


class XDdata(XDbase):
    """XDdata object represents a single xdphys data file
    
    It handles parsing of the file, section-wise.
    
    """

    # dtype of analog raster data (stimulus and trace):
    _RASTER_DTYPE = np.dtype("int16").newbyteorder('>')
    
    # Minimum supported xdphys version, use XDdata_legacy for older versions
    _XDPHYS_MIN_VER = '2.8.0'
    _XDPHYS_PROG_NAME = 'xdphys'

    # @see: XDbase._lazy_dict
    _lazy_dict = XDbase._lazy_dict.copy()

    def _parse_trials(self):
        with self._file as f:
            # seek RASTERDATA line
            f.seek(self._seek_rasterdata)
            # Next line must be "RASTERDATA":
            if not f.nextline() == "RASTERDATA":
                raise ValueError(f"RASTERDATA expected at position {self._seek_rasterdata}")
            # "nrasters=" line:
            l = f.nextline()
            if not l.startswith("nrasters="):
                raise ValueError(f"nrasters= expected in RASTERDATA at position {self._seek_rasterdata}")
            self.nrasters = int(l[9:]) # 9 == len('nrasters=')
            self.trials = []
            self.nevents = []
            self.events = []
            self._seek_ad = {}
            ### first "depvar=" line:
            l = f.nextline()
            for ktrial in range(self.nrasters):
                ### "depvar=" line (read before loop or at the end of last iteration):
                if l == "END_RASTERDATA" or l == '':
                    # No more trials, go back:
                    break
                t = OrderedDict(index = ktrial)
                m = self._depvar_re.fullmatch(l)
                if m:
                    t.update((k, self._convert_value(m.group(g))) for k, g in self._depvar_re_groups)
                elif l == 'depvar=-6666 <SPONT>':
                    # "spontaneous" trial have no params, fill with NaN:
                    # changed to leave them empty (pandas can handle this)
                    #t.update((k, np.nan) for k, g in self._depvar_re_groups)
                    t['stim'] = 'spont'
                else:
                    raise ValueError(f"Unable to parse depvar line: {l}")
                self.trials.append(t)
                ### "nevents=" line:
                l = f.nextline()
                if not l.startswith("nevents="):
                    raise ValueError(f"nevents= expected at position {f.tell() - len(l) -1}")
                self.nevents.append(int(l[8:])) # 8 == len('nevents=')
                events = []
                for kevent in range(self.nevents[-1]):
                    l = f.nextline()
                    events.append(tuple(int(e) for e in l.split("\t")))
                self.events.append(events)
                ### Analog data (TRACE or STIMULUS)
                # Read next line, can be one of 'STIMULUS', 'TRACE' or a "depvar=" line
                l = f.nextline()
                while l in ('STIMULUS', 'TRACE'):
                    ADKEY = l
                    adkey = ADKEY.lower()
                    adreadsize = getattr(self, f"_readsize_{adkey}")
                    l = f.nextline()
                    if not l.startswith("channel="):
                        raise ValueError(f"channel= expected at position {f.tell() - len(l) -1}")
                    adchannel = int(l[8:]) # 8 == len('channel=')
                    # Now we are exactly at the start of the ad data:
                    pos = f.tell()
                    self._seek_ad[(ktrial, adkey, adchannel)] = pos
                    # we skip reading it because we usually don't need to:
                    f.seek(pos + adreadsize + 1)
                    l = f.nextline()
                    if not l == f"END_{ADKEY}":
                        raise ValueError(f"END_{ADKEY} expected at position {f.tell() - len(l) -1}")
                    # Read the next 'STIMULUS', 'TRACE' or a "depvar=" line:
                    l = f.nextline()
            self.ntrials = len(self.trials)
    # Attributes made available by _parse_trials():
    _lazy_dict['trials'] = "_parse_trials"
    _lazy_dict['ntrials'] = "_parse_trials"
    _lazy_dict['nrasters'] = "_parse_trials"
    _lazy_dict['nevents'] = "_parse_trials"
    _lazy_dict['events'] = "_parse_trials"
    _lazy_dict['_seek_ad'] = "_parse_trials"
    
    def _parse_ad(self, adkey = None):
        """
        """
        if adkey is None:
            self._parse_ad(adkey = 'stimulus')
            self._parse_ad(adkey = 'trace')
            return
        assert adkey in ('stimulus', 'trace')
        channels = getattr(self, f"{adkey}_channels")
        arr = np.full(shape = (self.ntrials, getattr(self, f"{adkey}_len"), len(channels)),
                      fill_value = np.nan, dtype = self._RASTER_DTYPE.newbyteorder('='))
        readsize = getattr(self, f"_readsize_{adkey}")
        with self._file as f:
            for ktrial in range(self.ntrials):
                for kchan, chan in enumerate(channels):
                    f.seek(self._seek_ad[(ktrial, adkey, chan)])
                    r = f.read(readsize)
                    if isinstance(r, bytes):
                        r = r.decode('ascii')
                    arr[ktrial, :, kchan] = np.frombuffer(
                        bytes.fromhex(r.replace('\n', '')),
                        dtype=self._RASTER_DTYPE) #.astype('int16')
        setattr(self, adkey, arr)
        
    def _parse_stimulus(self):
        self._parse_ad('stimulus')
    _lazy_dict['stimulus'] = "_parse_stimulus"

    def _parse_trace(self):
        self._parse_ad('trace')
    _lazy_dict['trace'] = "_parse_trace"

    def _ad_sizes(self):
        """"""
        hexlen = 2 * self._RASTER_DTYPE.itemsize
        lam = lambda m: m + (m-1) // 80
        # TRACE:
        self.trace_len = int(1e-3 * self.params['epoch'] * self.params['adfc'] / self.params['ana']['decimate'])
        self._readsize_trace = lam(hexlen * self.trace_len)
        self.trace_channels = [k for k in (1, 2) if self.params['ana'][f'channel{k}'] == 'on']
        # STIMULUS:
        self.stimulus_len = int(1e-3 * self.params['epoch'] * self.params['dafc'] / self.params['stim']['decimate'])
        self._readsize_stimulus = lam(hexlen * self.stimulus_len)
        self.stimulus_channels = [1, 2] if self.params['stim']['save'] == 1 else []
    # Attributes made available by _ad_sizes():
    _lazy_dict['stimulus_len'] = "_ad_sizes"
    _lazy_dict['stimulus_channels'] = "_ad_sizes"
    _lazy_dict['trace_len'] = "_ad_sizes"
    _lazy_dict['trace_channels'] = "_ad_sizes"
    _lazy_dict['_readsize_stimulus'] = "_ad_sizes"
    _lazy_dict['_readsize_trace'] = "_ad_sizes"

    def _depvar_mapping(self):
        """"""
        # All newer versions (seen "2.7xx" and "2.8xx") use 
        gen_vars = ['abi', 'itd', 'iid', 'bc', 'stim', 'mono', 'two_snd']
        partial_depvar = "; ".join(["([^; ]+)"] * len(gen_vars))
        self._depvar_re = re.compile("depvar= ?([-0-9]+) <{}>".format(partial_depvar)) # for .gen files: "depvar=([0-9]+) <{}>" would be sufficient
        self._depvar_re_groups = tuple((gen_vars[k], k+2) for k in range(len(gen_vars)))
    # Attributes made available by _depvar_mapping():
    _lazy_dict['_depvar_re'] = "_depvar_mapping"
    _lazy_dict['_depvar_re_groups'] = "_depvar_mapping"
    
    def planned_trials(self):
        return self.params['reps'] * np.prod([
            np.asarray(v).size for v in self.params[self.fileinfo.mod].values()
        ])


class XDdata_legacy(XDdata):
    """Subclass of XDdata to handle specialties from older xdphys versions.
    
    For all public API documentation, see XDdata.
    
    Versions seen and supported:
    
    samplelength in the following description is calculated as:
    
        int(_adfc_ * _epoch_ * 1e-3 / _ana_decimate_)
    
    where _adfc_ is the sampling frequency in Hz (int) either from the ana-file
    or from the PARAMS section, _epoch_ is the recording length in milliseconds,
    _ana_decimate_ is the decimation factor used to reduce data, effectively
    reducing the sampling frequency by the this factor.
    
    == v2.47 before June 1999 (exclusive) ==
    
    * "depvar"-lines are "mod"-specific
    * contains no analog data in the same files
    * MAY have an ana-file (.ana or .ana.gz) containing one analog trace channel
    * ana files start with b'W1' and contain chunks of data for every trial:
      - 512 bytes "header", repeated for every trial
      - 2 * samplelength bytes analog trace (binary, big endian int16)
        
    == v2.47 between June and July 1999 (both inclusive) ==
    
    * "depvar"-lines include five fields: 'abi', 'itd', 'iid', 'bc', 'stim'
    * contains analog data as HEX values, but not surrounded by TRACE/END_TRACE
    * MAY have an ana-file (.ana or .ana.gz) containing one analog trace channel
    * ana files start with b'2W' and contain a global header followed by chunks
      of data for each trial:
      - first 4 bytes indicate version (?) b'2W', 3rd and 4th byte unknown
      - global header includes the following fields [bytelength] as
        null-terminated strings, each written as fieldname=value:
        + adFc [15]
        + epoch [15]
        + ana-decimate [15]
        + ana-tomv [30]
        + ana-offset [30]
        + nrasters [15]
        Total header length, including first 4 bytes is 124 bytes
      - For every trial, there is a chunk:
        + depvar [30], similar to the header fields
        + 2 * samplelength bytes analog trace (binary, little endian int16)
        
    A specialty of these files is that the traces are saved in two places. It
    seemed (in some examples) that these traces are identical. The HEX values
    will be used to fill the XDdata_legacy.trace values for these files.
    However, the _parse_anafile(), if called explicitly, will add another
    attribute XDdata_legacy.ana_trace which contains traces from the ana-file.
    It will also attempt to create an index list such that the following is
    true:
    
        # needed to make ana_trace and ana_ktrial available:
        xfile._parse_anafile()
        xfile.ana_trace[xfile.ana_ktrial, :, :] == xfile.trace
    
    == v2.47 after July 1999 (exclusive) and all higher versions ==
    
    Files are organized as in the most recent version (which is 2.8.1-1). The
    XDdata_legacy class is not needed for any of these versions as they can be
    handled with (slightly) higher performance by the XDdata base class.
    
    """
    # TODO: Check if there is a .ana file and implement reading it, see legacy_xdphys for how that works!
    
    # We have to copy the _lazy_dict to make changes of our own:
    _lazy_dict = XDdata._lazy_dict.copy()

    _XDPHYS_MIN_VER = '2.47'

    def __init__(self, filepath):
        super().__init__(filepath)
        # Eventually, we have a separate file with analog data:
        if self.ana_path is not None:
            self.has_anafile = True
            self._anafile = FileHandler(self.ana_path, mode='rb')
        else:
            self.has_anafile = False
        if self.fileinfo.ver >= super()._XDPHYS_MIN_VER:
            import warnings
            print(f'{self!s}, {self.fileinfo}')
            warnings.warn("For performance reasons, it is not advised to use"
                " XDdata_legacy for files also supported by XDdata.")

    def _find_anafile_path(self):
        fn = (lambda fn: fn[:-3] if fn.endswith('.gz') else fn)(self.path)
        if os.path.exists(fn + '.ana'):
            self.ana_path = fn + '.ana'
        elif os.path.exists(fn + '.ana.gz'):
            self.ana_path = fn + '.ana.gz'
        else:
            self.ana_path = None
    _lazy_dict['ana_path'] = '_find_anafile_path'

    def _legacy_version(self):
        if self.params['timestamp'] <= 928195200:
            # Versions before June 1999,
            # 928195200 is 1999-06-01T00:00:00+00:00
            self.v247_subversion = '1999-05'
        elif (self.params['timestamp'] > 928195200 and
                self.params['timestamp'] <= 933465600):
            # Versions in June and July 1999,
            # 928195200 is 1999-06-01T00:00:00+00:00
            # 933465600 is 1999-08-01T00:00:00+00:00
            self.v247_subversion = '1999-06'
        elif self.params['timestamp'] > 933465600:
            # This pattern was encountered after July 1999,
            # 933465600 is 1999-08-01T00:00:00+00:00
            self.v247_subversion = '1999-08'
        else:
            raise ValueError(f"Sorry, can't figure out how to read your xdphys file {self.path}")
    _lazy_dict['v247_subversion'] = '_legacy_version'
    
    def _depvar_mapping(self):
        """"""
        # Support for various patterns in old files (version 2.47)
        if self.fileinfo.ver > '2.47' or self.v247_subversion >= '1999-08':
            super()._depvar_mapping()
            return
        elif self.v247_subversion <= '1999-05':
            # These patterns were encountered in versions before June 1999,
            # At this time, patterns were specific to the "mod" used:
            if self.fileinfo.mod == 'itd':
                self._depvar_re = re.compile("depvar=([-0-9]+) <\\1 us>")
                self._depvar_re_groups = (('itd', 1),)
            elif self.fileinfo.mod == 'iid':
                self._depvar_re = re.compile("depvar=([-0-9]+) <\\1 db, L=[-0-9]+, R=[-0-9]+>")
                self._depvar_re_groups = (('iid', 1),)
            elif self.fileinfo.mod == 'bf':
                self._depvar_re = re.compile("depvar=([-0-9]+) <\\1 Hz>")
                self._depvar_re_groups = (('stim', 1),)
            elif self.fileinfo.mod == 'abi':
                self._depvar_re = re.compile("depvar=([-0-9]+) <\\1 dbspl>")
                self._depvar_re_groups = (('abi', 1),)
            elif self.fileinfo.mod == 'nop':
                self._depvar_re = re.compile("depvar=([-0-9]+) ;rep number")
                # It does contain the index (group 1), which we ignore:
                self._depvar_re_groups = () # (('index', 1),)
            elif self.fileinfo.mod == 'rov':
                self._depvar_re = re.compile("depvar=([-0-9]+) <([-0-9]+) us, ([-0-9]+) db> ;itd, iid")
                # It does contain some weird index number (group 1), which we ignore:
                self._depvar_re_groups = (('itd', 2), ('iid', 3)) # (('indexnum', 1), ('itd', 2), ('iid', 3))
            else:
                raise NotImplementedError(f"Unknown depvar mapping for xdphys file with version={self.fileinfo.ver} mod={self.fileinfo.mod}")
        elif self.v247_subversion == '1999-06':
            # This pattern was only encountered in June and July 1999
            # There were no 'mono' or 'two_snd' columns:
            gen_vars = ['abi', 'itd', 'iid', 'bc', 'stim']
            partial_depvar = "; ".join(["([^; ]+)"] * len(gen_vars))
            self._depvar_re = re.compile("depvar= ?([-0-9]+) <{}>".format(partial_depvar)) # for .gen files: "depvar=([0-9]+) <{}>" would be sufficient
            self._depvar_re_groups = tuple((gen_vars[k], k+2) for k in range(len(gen_vars)))
        else:
            raise NotImplementedError(f"Unknown depvar mapping for xdphys file with version={self.fileinfo.ver} mod={self.fileinfo.mod}")

    def _parse_trials(self):
        if self.fileinfo.ver >= super()._XDPHYS_MIN_VER:
            # Higher version's files are parsed with the XDdata function:
            super()._parse_trials()
            return
        # 
        with self._file as f:
            # seek RASTERDATA line
            f.seek(self._seek_rasterdata)
            # Next line must be "RASTERDATA":
            if not f.nextline() == "RASTERDATA":
                raise ValueError(f"RASTERDATA expected at position {self._seek_rasterdata}")
            # "nrasters=" line:
            l = f.nextline()
            if not l.startswith("nrasters="):
                raise ValueError(f"nrasters= expected in RASTERDATA at position {self._seek_rasterdata}")
            self.nrasters = int(l[9:]) # 9 == len('nrasters=')
            self.trials = []
            self.nevents = []
            self.events = []
            self._seek_ad = {}
            ### first "depvar=" line:
            l = f.nextline()
            for ktrial in range(self.nrasters):
                ### "depvar=" line (read before loop or at the end of last iteration):
                if l == "END_RASTERDATA" or l == '':
                    # No more trials, go back:
                    break
                t = OrderedDict(index = ktrial)
                m = self._depvar_re.fullmatch(l)
                if m:
                    t.update((k, self._convert_value(m.group(g))) for k, g in self._depvar_re_groups)
                elif l == 'depvar=-6666 <SPONT>':
                    # "spontaneous" trial have no params, fill with NaN:
                    # changed to leave them empty (pandas can handle this)
                    #t.update((k, np.nan) for k, g in self._depvar_re_groups)
                    t['stim'] = 'spont'
                else:
                    raise ValueError(f"Unable to parse depvar line: {l}")
                self.trials.append(t)
                ### "nevents=" line:
                l = f.nextline()
                if not l.startswith("nevents="):
                    raise ValueError(f"nevents= expected at position {f.tell() - len(l) -1}")
                self.nevents.append(int(l[8:])) # 8 == len('nevents=')
                events = []
                for kevent in range(self.nevents[-1]):
                    l = f.nextline()
                    events.append(tuple(int(e) for e in l.split("\t")))
                self.events.append(events)
                ### Analog data
                if self.fileinfo.ver > '2.47':
                    # Read next line, can be one of 'STIMULUS', 'TRACE' or a "depvar=" line
                    l = f.nextline()
                    while l in ('STIMULUS', 'TRACE'):
                        ADKEY = l
                        adkey = ADKEY.lower()
                        adreadsize = getattr(self, f"_readsize_{adkey}")
                        l = f.nextline()
                        if not l.startswith("channel="):
                            raise ValueError(f"channel= expected at position {f.tell() - len(l) -1}")
                        adchannel = int(l[8:]) # 8 == len('channel=')
                        # Now we are exactly at the start of the ad data:
                        pos = f.tell()
                        self._seek_ad[(ktrial, adkey, adchannel)] = pos
                        # we skip reading it because we usually don't need to:
                        f.seek(pos + adreadsize + 1)
                        l = f.nextline()
                        if not l == f"END_{ADKEY}":
                            raise ValueError(f"END_{ADKEY} expected at position {f.tell() - len(l) -1}")
                        # Read the next 'STIMULUS', 'TRACE' or a "depvar=" line:
                        l = f.nextline()
                else:
                    # For ver 2.47 file, there only is one trace saved in the
                    # same file - if there is one at all.
                    if not hasattr(self, '_v247_ad_in_file'):
                        # We have to check if trace data is saved by probing the
                        # first trial...
                        # Remember where it started:
                        pos = f.tell()
                        # read one line to see if there is analog data at all:
                        self._v247_ad_in_file = not f.nextline().startswith('depvar=')
                        # Go back to where we started probing:
                        f.seek(pos)
                    if self._v247_ad_in_file:
                        # Use default info, i.e. TRACE, channel=1
                        ADKEY = 'TRACE' # default value
                        adkey = ADKEY.lower()
                        adreadsize = getattr(self, f"_readsize_{adkey}")
                        adchannel = 1 # default value
                        # Now we are exactly at the start of the ad data:
                        pos = f.tell()
                        self._seek_ad[(ktrial, adkey, adchannel)] = pos
                        # we skip reading it because we usually don't need to:
                        f.seek(pos + adreadsize + 1)
                    # Read the next "depvar=" line:
                    l = f.nextline()
            self.ntrials = len(self.trials)

    def _parse_anafile(self):
        assert self.has_anafile, f"There is no corresponding ana-file for {self!s}"
        if self.v247_subversion <= '1999-05':
            with self._anafile as f:
                # Probe version:
                head_chunk = f.read(16)
                assert head_chunk[0:2] == b'W1', f"This ana-file version cannot currently be read {head_chunk}"
                chunksize = (512 + self._readsize_trace)
                # go back to start to read trial by trial:
                self.trace = np.full(shape = (self.ntrials, self.trace_len, len(self.trace_channels)),
                                 fill_value = np.nan,
                                 dtype = self._RASTER_DTYPE.newbyteorder('='))
                self.ana_trial_heads = []
                for ktrial in range(self.ntrials):
                    f.seek(ktrial * chunksize)
                    chunk = f.read(chunksize)
                    if not chunk[4:16] == head_chunk[4:16]:
                        raise ValueError("Error reading ana-file '{fn}' at trial #{kt}")
                    self.trace[ktrial, :, 0] = np.frombuffer(chunk[512:chunksize], dtype=self._RASTER_DTYPE)
                    self.ana_trial_heads.append(chunk[:512])
        elif self.v247_subversion == '1999-06':
            # This code will never run as part of a lazy-evaluation routine,
            # because files in question already have the analog data saved
            # internally as hex-values.
            # TODO: this is NOT yet finished!!!
            with self._anafile as f:
                # Header of four bytes:
                head_chunk = f.read(4)
                assert head_chunk[0:2] == b'2W', f"This ana-file version cannot currently be read {head_chunk}"
                # Global header with fields:
                fields = [('adFc', 15), ('epoch', 15), ('ana-decimate', 15), ('ana-tomv', 30), ('ana-offset', 30), ('nrasters', 15)]
                d = OrderedDict()
                for fieldname, fieldlength in fields:
                    fieldcontent = f.read(fieldlength).rstrip(b'\n\x00').decode('ascii')
                    assert fieldcontent.startswith(fieldname), f"Field {fieldname} expected, but got {fieldcontent}"
                    d[self._clean_key(fieldname)] = self._convert_value(fieldcontent[len(fieldname)+1:])
                # We already know all this, let's make sure it's correct:
                assert d['adfc'] == self.params['adfc'], 'Mismatch "adfc" between ana-file "{self.ana_path!s}" and params in {self!s}'
                assert d['epoch'] == self.params['epoch'], 'Mismatch "epoch" between ana-file "{self.ana_path!s}" and params in {self!s}'
                assert d['ana_decimate'] == self.params['ana']['decimate'], 'Mismatch "ana_decimate" between ana-file "{self.ana_path!s}" and params in {self!s}'
                assert d['ana_tomv'] == self.params['ana']['tomv'], 'Mismatch "ana_tomv" between ana-file "{self.ana_path!s}" and params in {self!s}'
                assert d['ana_offset'] == self.params['ana']['offset'], 'Mismatch "ana_offset" between ana-file "{self.ana_path!s}" and params in {self!s}'
                assert d['nrasters'] == self.ntrials, 'Mismatch between "nrasters" in ana-file "{self.ana_path!s}" and ntrials in {self!s}'
                # In order to align traces from ana-file and from HEX data,
                # figure out what depvar was used, and get the trials' values:
                depvar_key = self.params['depvar'].split(" ")[0]
                if not depvar_key == 'gen':
                    if depvar_key == 'bf':
                        depvar_key = 'stim'
                    tdepvars = [t[depvar_key] for t in self.trials]
                self.ana_ktrial = [None] * self.ntrials
                self.ana_trace = np.full(shape = (self.ntrials, self.trace_len, 1),
                                  fill_value = np.nan,
                                  dtype = self._RASTER_DTYPE.newbyteorder('='))
                for kana in range(self.ntrials):
                    depvarcontent = f.read(30).rstrip(b'\n\x00').decode('ascii')
                    depvar = self._convert_value(depvarcontent[7:]) # 7 == len('depvar')+1
                    self.ana_trace[kana,:,0] = np.fromfile(f, dtype=self._RASTER_DTYPE.newbyteorder('<'), count=self.trace_len)
                    # See if we can find the same in self.trace
                    if not depvar_key == 'gen':
                        try:
                            ktrial = tdepvars.index(depvar)
                            while (not np.all(self.trace[ktrial,:,0] == self.ana_trace[kana,:,0])):
                                ktrial = tdepvars.index(depvar, ktrial+1)
                            tdepvars[ktrial] = None
                            self.ana_ktrial[ktrial] = kana
                        except ValueError:
                            # This happens if no matching trace in self.trace was found:
                            pass
        else:
            raise NotImplementedError("xdphys versions after July 1999 are not know to have ana files.")

    def _ad_sizes(self):
        """"""
        if self.fileinfo.ver >= super()._XDPHYS_MIN_VER:
            # Higher version's files are parsed with the XDdata function:
            super()._ad_sizes()
            return
        # version 2.7x support:
        if self.fileinfo.ver >= '2.7':
            hexlen = 2 * self._RASTER_DTYPE.itemsize
            lam = lambda m: m + (m-1) // 80
            # TRACE:
            self.trace_len = int(1e-3 * self.params['epoch'] * self.params['adfc'] / self.params['ana']['decimate'])
            self._readsize_trace = lam(hexlen * self.trace_len)
            self.trace_channels = [k for k in (1, 2) if self.params['ana'][f'channel{k}'] == 'on']
            # No stimulus in v2.7x files:
            self.stimulus_len = int(1e-3 * self.params['epoch'] * self.params['dafc'])
            self.stimulus_channels = []
            self._readsize_stimulus = 0
            return
        # version 2.47 support:
        hexlen = 2 * self._RASTER_DTYPE.itemsize
        lam = lambda m: m + (m-1) // 80
        if self.v247_subversion <= '1999-05':
            self.params['ana'] = OrderedDict()
            if self.has_anafile:
                with self._anafile as f:
                    # Get information from ana-file that is usually in PARAMS
                    head_chunk = f.read(16)
                    assert head_chunk[0:2] == b'W1', f"Unexpected ana-file format."
                    samplingrate, samplelength, chan = np.frombuffer(head_chunk[4:16], dtype=np.dtype('int32').newbyteorder('>'))
                    self.trace_len = int(samplelength)
                    self.trace_channels = [int(chan)]
                    # For reading binary data, _not_ HEX values:
                    self._readsize_trace = self._RASTER_DTYPE.itemsize * self.trace_len
                    self.params['ana']['save'] = 1
                    self.params['ana']['decimate'] = int(self.params['adfc']/samplingrate)
                    # TODO: do we have to guess the following?
                    self.params['ana']['every'] = 1
                    self.params['ana']['tomv'] = 0.306523
                    self.params['ana']['offset'] = 0.000000
            else:
                # These values are "made up", but should somehow work as they are
                self.trace_len = int(1e-3 * self.params['epoch'] * self.params['adfc'])
                self.trace_channels = []
                self._readsize_trace = 0
                self.params['ana']['save'] = 0
        elif self.v247_subversion >= '1999-06':
            # This works for the short lived mid-1999 and later versions
            self.trace_len = int(1e-3 * self.params['epoch'] * self.params['adfc'] / self.params['ana']['decimate'])
            self.trace_channels = [1] if self.params['ana']['save'] == 1 else []
            self._readsize_trace = lam(hexlen * self.trace_len)
        # All of the 2.47 file have no stimulus:
        self.stimulus_len = int(1e-3 * self.params['epoch'] * self.params['dafc'])
        self.stimulus_channels = []
        self._readsize_stimulus = 0
    
    def _parse_params(self):
        super()._parse_params()
        if self.v247_subversion <= '1999-05':
            # In these files, we only get all the information from probing
            # the ana-file (if one exists). This will extend self.params:
            self._ad_sizes()
    
    def _parse_ad(self, adkey = 'trace'):
        if self.v247_subversion >= '1999-06':
            super()._parse_ad(adkey)
        elif adkey == 'trace' and self.has_anafile:
            self._parse_anafile()
        else:
            if adkey == 'trace':
                self.trace = np.full(shape = (self.ntrials, self.trace_len, len(self.trace_channels)),
                                 fill_value = np.nan,
                                 dtype = self._RASTER_DTYPE.newbyteorder('='))
            elif adkey == 'stimulus':
                self.stimulus = np.full(shape = (self.ntrials, self.stimulus_len, len(self.stimulus_channels)),
                                 fill_value = np.nan,
                                 dtype = self._RASTER_DTYPE.newbyteorder('='))


class XDcal(XDbase):
    """XDcal object represents a single xdphys calibration file
    """

    # Minimum supported xdphys version, use XDdata_legacy for older versions
    _XDPHYS_MIN_VER = '2.8.0'
    _XDPHYS_PROG_NAME = 'xcalibur'

    # @see: XDbase._lazy_dict
    _lazy_dict = XDbase._lazy_dict.copy()

    def _parse_caldata(self):
        with self._file as f:
            # seek RASTERDATA line
            f.seek(self._seek_rasterdata)
            # Next line must be "RASTERDATA":
            if not f.nextline() == "RASTERDATA":
                raise ValueError(f"RASTERDATA expected at position {self._seek_rasterdata}")
            self.raster_comments = []
            while True:
                # "nrasters=" line:
                l = f.nextline()
                if l.startswith("; "):
                    self.raster_comments.append(l.strip("; "))
                    continue
                if not l.startswith("nrasters="):
                    raise ValueError(f"nrasters= expected in RASTERDATA at position {self._seek_rasterdata}")
                self.nrasters = int(l[9:]) # 9 == len('nrasters=')
                break
            self.calcomments = {}
            for cmt in self.raster_comments:
                m = re.fullmatch("([0-9]+): ([a-z_]+)(?: ?(.*))?", cmt)
                if m is None:
                    raise ValueError(f"Unknown comment: '{cmt}'")
                colnum, varname, description = m.groups()
                self.calcomments[int(colnum)] = (varname, description)
            self.caldata_wrapped = pd.DataFrame(
                np.loadtxt([f.nextline() for k in range(self.nrasters)]),
                columns=[self.calcomments[kc][0] for kc in sorted(self.calcomments.keys())],
            ).set_index("freq")
            f_seek_endpos = f.tell()
            l = f.nextline()
            if not l == "END_RASTERDATA":
                raise ValueError(f"END_RASTERDATA expected at position {f_seek_endpos}")
            # Finally, unwrap '_phase' columns:
            self.caldata = self.caldata_wrapped.copy()
            for c in self.caldata.columns[self.caldata.columns.str.endswith('_phase')]:
                self.caldata[c] = np.unwrap(self.caldata[c])
    _lazy_dict['nrasters'] = "_parse_caldata"
    _lazy_dict['raster_comments'] = "_parse_caldata"
    _lazy_dict['caldata_wrapped'] = "_parse_caldata"
    _lazy_dict['caldata'] = "_parse_caldata"

    def interpolate_caldata(self, freqs, interpolate_phase_as_delay = False, caldata = None):
        """Return linearly interpolated caldata for use in calibration/decalibration.
        
        To be used for transforming a stimulus (calibration or decalibration), the given
        caldata must be interpolated to provide values for every (dFFT) frequency that shall
        be affected by the transformation. In xdphys (??) and BEDS, this interpolation is
        done linearly between frequencies present in the original caldata.
        """
        
        if caldata is None:
            caldata = self.caldata
        
        caldata_inter = pd.DataFrame([],
            columns=caldata.columns,
            index=pd.Series(freqs, name=caldata.index.name)
        )

        for c in caldata_inter.columns:
            if interpolate_phase_as_delay and c.endswith('_phase'):
                # This was done in BEDS (don't know about xdphys yet) but the effect is minimal
                # Converts phases to delays (in seconds), then interpolates, then converts back
                caldata_inter[c] = (2 * np.pi * freqs) * np.interp(freqs, caldata.index, caldata[c] / (2 * np.pi * caldata.index), left = np.nan, right = np.nan)
            else:
                caldata_inter[c] = np.interp(freqs, caldata.index, caldata[c], left = np.nan, right = np.nan)
        return caldata_inter


class FileHandler():
    """A context handler class that keeps track of how often it is used
    """
    def __init__(self, path, mode='r'):
        self.path = path
        self.mode = mode
        self.handle_count = 0

    def __enter__(self):
        if self.handle_count == 0:
            if self.path.endswith('.gz'):
                self.file_obj = gzip.open(self.path, self.mode)
            else:
                self.file_obj = open(self.path, self.mode)
            # Read a small chunk to see if we'll get bytes or str, then
            # bind appropriate nextline function:
            chunk = self.file_obj.read(1)
            # Set pointer back to first byte
            self.file_obj.seek(0)
            if isinstance(chunk, str):
                self.file_obj.nextline = MethodType(self._nextline_str, self.file_obj)
            else:
                self.file_obj.nextline = MethodType(self._nextline_bytes, self.file_obj)
        self.handle_count += 1
        return self.file_obj

    def __exit__(self, errtype, value, traceback):
        self.handle_count -= 1
        if self.handle_count == 0:
            self.file_obj.close()
            del self.file_obj

    @staticmethod
    def _nextline_str(file_obj):
        """Read the nextline of file_obj. Removes trailing newlines."""
        return file_obj.readline().rstrip('\r\n')
    @staticmethod
    def _nextline_bytes(file_obj):
        """Read the nextline of file_obj. Removes trailing newlines."""
        return file_obj.readline().decode('ascii').rstrip('\r\n')

