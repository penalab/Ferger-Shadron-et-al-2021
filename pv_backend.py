"""Module to prepate and organize data for PV paper figures
"""

# Builtin
import sys
import os
import time
import glob
from collections import namedtuple

# 3rd party
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition

# Data specific "manager" functions
from pedman import * # Site
# LazyLoader metaclass to load attributes dynamically as needed
from pedman._internals import LazyLoader
# PV model from Fischer&Pena 2011
from pv_model import PVmodelRunner

# Enable file based caching of some "expensive" to load variables:
from pedman.cache import PedmanCache
PedmanCache()

class PV_DataContainer(metaclass=LazyLoader):
    """Handling class to contain data (original and analyzed)
    Mainly to keep analysis and visualization separated and allow easier
    compilation of combined figures showing the results of different
    analysis steps.
    
    This class uses a lazy loading interface (see LazyLoader for implementation
    details) to make attributes easily accessible, but computed on demand.
    """
    
    def __init__(self,
        meta_files_dir = None,
        basedir = None,
        meta_file_pattern = "site*.yaml",
        **kwargs
    ):
        """PV_DataContainer
        
        Arguments
        ---------
        meta_files_dir : <path>
            Path to the directory where meta files are found.
        basedir : <path>
            Path to the base directory (see pedman.Site for details)
        meta_file_pattern : str
            Glob filename pattern for meta files
        """
        self.meta_files_dir = meta_files_dir
        self.basedir = basedir
        self.meta_file_pattern = meta_file_pattern
        self.options = kwargs
        
    def _sites(self):
        """Load and filter sites
        
        Also 'reset' the stimulus duration of itd_pairs_bc recordings.
        """
        self.sites = []
        for mf in glob.glob(os.path.join(self.meta_files_dir, self.meta_file_pattern)):
            s = Site(mf, basedir = self.basedir)
            if 'population vector' in s.meta.get('review', {}).get('good_for', []):
                self.sites.append(s)
        # Set analysis window (duration to count spike times) for all
        # 'itd_pairs_bc' recordings to a duration of 100 ms to make it
        # comparable with the "normal" itd tunings and allow
        # normalization in relation to the latter.
        print("The 'dur' of all 'itd_pairs_bc' recordings was artificially"
              "set to 100 [ms], to allow normalization.",
              file=sys.stderr, flush=True)
        for s in self.sites:
            for r in s.recs[('itd_pairs_bc')]:
                r.xfile.params['dur'] = 100
    _lazy['sites'] = '_sites'
    
    def _sites_overview_data(self):
        site_tuple = namedtuple('SiteDataTuple', 
            's sid n_units best_itds u_colors hemisphere_L best_itd_range max_responses best_freqs min_freqs max_freqs',
            defaults = [None] * 11)
        self.sites_overview_data = []
        for ks, s in enumerate(self.sites):
            best_itds = [u.get_best_itd() for u in s.units.values()]
            best_itd_range = np.ptp(best_itds)
            try:
                best_freqs = [u.get_best_freq() for u in s.units.values()]
                min_freqs = [u.get_min_freq() for u in s.units.values()]
                max_freqs = [u.get_max_freq() for u in s.units.values()]
            except:
                best_freqs = None
                min_freqs = None
                max_freqs = None
            self.sites_overview_data.append(
                site_tuple(
                    s = s,
                    sid = s.id(),
                    n_units = len(s.units),
                    best_itds = best_itds,
                    u_colors = [float(u.channel()) for u in s.units.values()],
                    hemisphere_L = bool(s.meta['position'].get('left', False)),
                    best_itd_range = np.ptp(best_itds),
                    max_responses = [u.get_max_response() for u in s.units.values()],
                    best_freqs = best_freqs,
                    min_freqs = min_freqs,
                    max_freqs = max_freqs,
                )
            )
    _lazy['sites_overview_data'] = '_sites_overview_data'
    
    def _all_units(self):
        self.all_units = [u for s in self.sites for u in s.units.values()]
    _lazy['all_units'] = '_all_units'
    
    def _spreading_responses(self):
        rr_all = []
        for u in self.all_units:
            fileindex = u.site.recs['itd_pairs_bc', 0].index
            
            responses =  u.responses(fileindex).copy()
            
            if self.options.get('scale_norm', False):
                responses *= u.responses_scale_factor(fileindex, by_bc=100)
            
            resp_100 = responses.query('bc == 100')
            resp_40 = responses.query('bc == 40')
            resp_20 = responses.query('bc == 20')
            
            norm_resp_100 = resp_100 / u.get_max_response(bc = 100)
            norm_resp_40 = resp_40 / u.get_max_response(bc = 40)
            norm_resp_20 = resp_20 / u.get_max_response(bc = 20)
            norm100_resp_40 = resp_40 / u.get_max_response(bc = 100)
            norm100_resp_20 = resp_20 / u.get_max_response(bc = 100)
            
            stim_itds = u.site.recs[fileindex].xfile.params['gen']['itd']
            best_itd = u.get_best_itd()
            
            mean_resp_100 = resp_100.groupby('itd').mean()
            mean_resp_40 = resp_40.groupby('itd').mean()
            mean_resp_20 = resp_20.groupby('itd').mean()
            mean_norm_resp_100 = norm_resp_100.groupby('itd').mean()
            mean_norm_resp_40 = norm_resp_40.groupby('itd').mean()
            mean_norm_resp_20 = norm_resp_20.groupby('itd').mean()
            mean_norm100_resp_40 = norm100_resp_40.groupby('itd').mean()
            mean_norm100_resp_20 = norm100_resp_20.groupby('itd').mean()
            
            for stim_itd in stim_itds:
                rr_40 = (mean_resp_40.loc[stim_itd] / mean_resp_100.loc[stim_itd]).iat[0]
                norm_rr_40 = (mean_norm_resp_40.loc[stim_itd] / mean_norm_resp_100.loc[stim_itd]).iat[0]
                rr_20 = (mean_resp_20.loc[stim_itd] / mean_resp_100.loc[stim_itd]).iat[0]
                norm_rr_20 = (mean_norm_resp_20.loc[stim_itd] / mean_norm_resp_100.loc[stim_itd]).iat[0]
                rr_all.append({
                    'u': u,
                    'best_itd': best_itd,
                    'stim_itd': stim_itd,
                    'delta_itd': np.abs(stim_itd-best_itd),
                    'delta_itd_signed': (stim_itd-best_itd),
                    'mean_resp_100': mean_resp_100.loc[stim_itd].iat[0],
                    'mean_resp_40': mean_resp_40.loc[stim_itd].iat[0],
                    'mean_resp_20': mean_resp_20.loc[stim_itd].iat[0],
                    'mean_norm_resp_100': mean_norm_resp_100.loc[stim_itd].iat[0],
                    'mean_norm_resp_40': mean_norm_resp_40.loc[stim_itd].iat[0],
                    'mean_norm_resp_20': mean_norm_resp_20.loc[stim_itd].iat[0],
                    'mean_norm100_resp_40': mean_norm100_resp_40.loc[stim_itd].iat[0],
                    'mean_norm100_resp_20': mean_norm100_resp_20.loc[stim_itd].iat[0],
                    'rr_40': rr_40,
                    'rr_20': rr_20,
                    # 'ratio_20_40': rr_20/rr_40 if rr_40 != 0 else np.nan,
                    'norm_rr_40': norm_rr_40,
                    'norm_rr_20': norm_rr_20,
                    # 'norm_ratio_20_40': norm_rr_20/norm_rr_40 if norm_rr_40 != 0 else np.nan,
                })
        df = pd.DataFrame(rr_all)
        df['ku'] = df['u'].apply(str)
        df.set_index(keys='ku', append=False, drop=True, inplace=True)
        self.spreading_df = df
    _lazy['spreading_df'] = '_spreading_responses'
    
    def _spreading_fit(self):
    
        # Allowed values:
        fit_params = self.options.get('spread_fit_params', 'sd')
        if not fit_params in ('sd', 'sd_scale'):
            raise ValueError("Only 'sd' or 'sd_scale' allowed for spread_fit_params.")
    
        spreading_fit = {}

        spreading_fit['_scaled'] = self.options.get('scale_model_profile', False)
        
        for bc in [100, 40, 20]:
            x = -self.spreading_df['delta_itd_signed'].values
            y = self.spreading_df[f'mean_norm_resp_{bc}'].values
            model_prof = PopulationProfile(stim_itd=0, bc=bc)
        
            if spreading_fit['_scaled']:
                prof_scale = optimal_scale(model_prof, x, y)
            else:
                prof_scale = 1.0

            def make_error_function(myx, myy, scale=1.0, bc = bc):
                # Create PopulationProfile to be used in error function
                # The stim_itd will be *changed* during optimization, don't
                # use this object later. Initializing with actual stim_itd:
                p = PopulationProfile(stim_itd = 0, bc = bc)
                def error_fun(x):
                    """Error function to minimize with scipy.optimize.minimize"""
                    # change p.stim_itd, this is what's being optimized:
                    p.w = x[0]
                    if fit_params == 'sd_scale':
                        myscale = x[1]
                    else:
                        myscale = scale
                    return np.sqrt(
                        np.mean(
                            (myscale * p(myx) - myy) ** 2
                        )
                    )
                return error_fun

            err_fun = make_error_function(x, y, scale=prof_scale, bc = bc)
            
            if fit_params == 'sd':
                minimize_result = sp.optimize.minimize(
                    err_fun,
                    (model_prof.w, ),
                )
                spreading_fit[bc] = {
                    'sd': minimize_result['x'][0],
                    'scale': prof_scale,
                    'model_scale': prof_scale,
                    'minimize_result': minimize_result,
                }
            else: # fit_params == 'sd_scale'
                minimize_result = sp.optimize.minimize(
                    err_fun,
                    (model_prof.w, prof_scale),
                    bounds = [(None, None), (0, 1.5)],
                )
                spreading_fit[bc] = {
                    'sd': minimize_result['x'][0],
                    'scale': minimize_result['x'][1],
                    'model_scale': prof_scale,
                    'minimize_result': minimize_result,
                }
            spreading_fit[bc]['model_profile'] = model_prof
            prof_opt = PopulationProfile(
                stim_itd = 0, fixed_w = spreading_fit[bc]['sd'],
            )
            spreading_fit[bc]['optimal_profile'] = prof_opt
            spreading_fit[bc]['model_rmse'] = np.sqrt(np.mean((
                spreading_fit[bc]['model_scale'] * model_prof(x) - y
            ) ** 2))
            spreading_fit[bc]['model_rsquare'] = 1 - np.sum((
                y - spreading_fit[bc]['model_scale'] * model_prof(x)
            ) ** 2) / np.sum((
                y - np.mean(y)
            ) ** 2)
            spreading_fit[bc]['optimal_rmse'] = np.sqrt(np.mean((
                spreading_fit[bc]['scale'] * prof_opt(x) - y
            ) ** 2))
            spreading_fit[bc]['optimal_rsquare'] = 1 - np.sum((
                y - spreading_fit[bc]['scale'] * prof_opt(x)
            ) ** 2) / np.sum((
                y - np.mean(y)
            ) ** 2)
        
        self.spreading_fit = spreading_fit

    _lazy['spreading_fit'] = '_spreading_fit'

    def _run_pv_models(self):
        # PV model with 2011 standard deviations:
        # re-seed RandomState for reproducible results
        # (default integers drawn on 2021-05-15)
        np.random.seed(self.options.get('seed_pv_mod2011', 535086493))
        self.pv_mod2011 = PVmodelRunner()
        # PV model with standard deviations fitted to present data:
        # re-seed RandomState for reproducible results
        # (default integers drawn on 2021-05-15)
        np.random.seed(self.options.get('seed_pv_mod2011', 1943141217))
        bcs = [20, 40, 100]
        sds = [self.spreading_fit[bc]['sd'] for bc in bcs]
        self.pv_fit2021 = PVmodelRunner(bcs = bcs, sds = sds)
    _lazy['pv_mod2011'] = '_run_pv_models'
    _lazy['pv_fit2021'] = '_run_pv_models'

    def _default_itds(self):
        # ITDs: -240, -220, -200, ..., -20, 0, +20, ..., +200, +220, +240
        self.default_itds = np.linspace(-240,240, 25).tolist()
        self.default_itds_set = set(self.default_itds)
    _lazy['default_itds'] = '_default_itds'
    _lazy['default_itds_set'] = '_default_itds'

    def _itd_sites(self):
        """Find sites with all default_itds in itd_bc recording"""
        self.itd_sites = []

        for s in self.sites:
            try:
                r = s.recs['itd_bc', 0]
            except:
                print(s, "No itd_bc")
                continue
            s_itds = r.xfile.params['gen']['itd'].tolist()
            if s_itds == self.default_itds:
                # Stimulus ITDs exactly match default_itds:
                self.itd_sites.append(s)
            elif set(s_itds).issuperset(self.default_itds_set):
                # Stimulus ITDs are a superset of default_itds:
                self.itd_sites.append(s)
            else:
                # no compatible ITD stimulus range
                pass
    _lazy['itd_sites'] = '_itd_sites'
    
    def _responses_stimitd_bestitd(self, bc = 100):
        """"""
        if not 'spreading2_bins' in self.__dict__:
            self.spreading2_bins = np.linspace(-130,130,14)
        out_list = []
        for s in self.itd_sites:
            r = s.recs['itd_bc', 0]
            max_responses = pd.Series((u.get_max_response() for u in s.units.values()), index=s.units.keys())
            resp_bc = r.responses().xs(bc, level='bc')
            resp = resp_bc.groupby('itd').agg(['mean', 'sem'])
            resp_norm = resp_bc.divide(max_responses).groupby('itd').agg(['mean', 'sem'])

            for u in s.units.values():
                for stim_itd in self.default_itds:
                    out_list.append(dict(
                        sid = s.id(),
                        uname = u.name,
                        best_itd = u.get_best_itd(),
                        stim_itd = stim_itd,
                        resp_all = resp_bc.query('itd == @stim_itd')[u.name].values,
                        resp = resp.loc[stim_itd, (u.name, 'mean')],
                        resp_norm = resp_norm.loc[stim_itd, (u.name, 'mean')],
                        resp_sem = resp.loc[stim_itd, (u.name, 'sem')],
                        resp_norm_sem = resp_norm.loc[stim_itd, (u.name, 'sem')],
                    ))
        out = pd.DataFrame(out_list)
        setattr(self, 'spreading2_out' + ('' if bc == 100 else f'_bc{bc}'), out)
        
        out_vals_list = []
        for stim_itd in self.default_itds:
            out_stim = out.query('stim_itd == @stim_itd')
            out_vals_list.append(out_stim.groupby(pd.cut(out_stim["best_itd"], bins = self.spreading2_bins)).agg({'resp_norm': 'mean'}).values)
            
        spreading2_out_vals = np.array(out_vals_list).squeeze().T
        setattr(self, 'spreading2_out_vals' + ('' if bc == 100 else f'_bc{bc}'), spreading2_out_vals)
    _lazy['spreading2_bins'] = '_responses_stimitd_bestitd'
    _lazy['spreading2_out'] = '_responses_stimitd_bestitd'
    _lazy['spreading2_out_vals'] = '_responses_stimitd_bestitd'
    
    def _responses_stimitd_bestitd_bc40(self):
        self._responses_stimitd_bestitd(bc = 40)
    _lazy['spreading2_out_bc40'] = '_responses_stimitd_bestitd_bc40'
    _lazy['spreading2_out_vals_bc40'] = '_responses_stimitd_bestitd_bc40'
    def _responses_stimitd_bestitd_bc20(self):
        self._responses_stimitd_bestitd(bc = 20)
    _lazy['spreading2_out_bc20'] = '_responses_stimitd_bestitd_bc20'
    _lazy['spreading2_out_vals_bc20'] = '_responses_stimitd_bestitd_bc20'
    
    def population_density_at(self, itd):
        """Return """
        # Standard deviation (from Fischer&Pena 2011):
        sigma_deg = 23.3 # in degree
        sigma = sigma_deg * 2.8 # microseconds or "ITD"
        return (1 / (sigma * np.sqrt(2 * np.pi)) ) * np.exp(- 0.5 * (itd ** 2 / sigma ** 2))
    
    def reset_exemplary_site(self, site_id = "20190404-01"):
        if 'exemplary_site' in self.__dict__:
            if self.exemplary_site.id() == site_id:
                # already set correctly
                return
            else:
                # clear data that depends on exemplary_site:
                for k, v in self._lazy.items():
                    if v in ('_exemplary_itd_tuning'):
                        self.__dict__.pop(k, None)
        for s in self.sites:
            if s.id() == site_id:
                self.exemplary_site = s
                break
        else:
            raise ValueError(f"No Site with ID '{site_id}' found.")
        # Set showcase ITDs:
        self._exemplary_showcase_stim_itds()
    _lazy['exemplary_site'] = 'reset_exemplary_site'

    def _exemplary_showcase_stim_itds(self, reset = False):
        s = self.exemplary_site
        if not reset and hasattr(s, 'showcase_stim_itds'):
            # already has a showcase_stim_itds attribute
            return
        r = s.recs['itd_pairs_bc', 0]
        stim_itds = sorted(r.xfile.params['gen']['itd'])
        if len(stim_itds) == 4:
            # We're good, 4 ITDs is what we want to show
            s.showcase_stim_itds = stim_itds
        elif len(stim_itds) == 6:
            # Show the two most extreme and the two central ITDs:
            s.showcase_stim_itds = sorted(
                [stim_itds[0], stim_itds[2], stim_itds[3], stim_itds[5]]
            )
        else:
            # We don't know how to make this
            raise ValueError("Not sure which stim_itds to show."
                f"Please select a subset of {stim_itds}, and assign "
                "as list to pvdc.exemplary_site.showcase_stim_itds"
            )

    def _exemplary_itd_tuning(self, bc = 100):
        if bc == 100:
            _bc = ''
        elif bc in (20, 40):
            _bc = f'_bc{bc}'
        else:
            raise ValueError('BC must be 20, 40 or 100')

        s = self.exemplary_site
        
        r = s.recs["itd_bc", 0]
        resp = r.responses().groupby(['bc', 'itd']).agg(['mean', 'sem'])
        spont = r.spontrates().mean()
        self.exemplary_itd_resp = resp
        self.exemplary_itd_spont = spont
        self.exemplary_itd_units = s.units
    _lazy['exemplary_itd_resp'] = '_exemplary_itd_tuning'
    _lazy['exemplary_itd_spont'] = '_exemplary_itd_tuning'
    _lazy['exemplary_itd_units'] = '_exemplary_itd_tuning'
    
    def _pv_decode(self):
        sites_estimators = []
        for s in self.sites:

            r = s.recs['itd_pairs_bc', 0]

            best_itds = np.asarray([u.get_best_itd() for u in s.units.values()])
            resp = r.responses().copy() #.divide(max_responses)
            bcs = np.unique(resp.index.get_level_values('bc'))
            stim_itds = np.asarray(np.unique(resp.index.get_level_values('itd')))

            max_responses = {
                bc: np.array([u.get_max_response(bc=bc) for u in s.units.values()])
                for bc in bcs
            }
            
            if self.options.get('scale_norm', False):
                for u in s.units.values():
                    resp[u.name] *= u.responses_scale_factor(r.index, by_bc=100)

            norm_itds = normalize_stim_itds(best_itds, stim_itds)

            for kstim, stim_itd in enumerate(stim_itds):

                for kbc, bc in enumerate(bcs[::-1]):
                    # Create and run ITD Estimator:
                    estimator = PVdecoder(
                        xvalues = best_itds,
                        yvalues = resp.xs([stim_itd, bc], level=['itd', 'bc']).values / max_responses[bc],
                        stim_itd = stim_itd,
                        #bc = bc,
                        fixed_w = self.spreading_fit[bc]['sd'],
                        # scale = None, # None: Optimize scale of population profile function to mean responses before fitting
                        # scale = 1.0, # 1: Do NOT optimize scale
                        scale = self.options.get('decoder_scale', None)
                    )
                    eligible = np.mean(estimator.y, axis=1) > 0.15
                    sites_estimators.append({
                        'site': s,
                        'rec': r,
                        'best_itds': best_itds,
                        'stim_itd': stim_itd,
                        'norm_itd': norm_itds[kstim],
                        'bc': bc,
                        'eligible': eligible,
                        'estimator': estimator,
                        'estimator_itd': estimator.itd,
                        'estimator_itds_all': estimator.itds,
                        'estimator_itds': estimator.itds[eligible],
                        'estimator_x': estimator.x,
                        'estimator_y': estimator.y,
                        'estimator_y_mean': estimator.y_mean,
                        'delta_itd': estimator.itd - stim_itd,
                        'delta_itds_all': estimator.itds - stim_itd,
                        'delta_itds': estimator.itds[eligible] - stim_itd,
                        'estimation_err': np.sqrt(np.mean((estimator.itds[eligible] - stim_itd) ** 2)),
                        'estimation_err_all': np.sqrt(np.mean((estimator.itds - stim_itd) ** 2)),
                        'estimation_var': np.var(estimator.itds[eligible] - stim_itd),
                        'estimation_var_all': np.var(estimator.itds - stim_itd),
                        'estimation_sem': sp.stats.sem(estimator.itds[eligible] - stim_itd),
                        'estimation_sem_all': sp.stats.sem(estimator.itds - stim_itd),
                        'estimation_std': np.std(estimator.itds[eligible] - stim_itd),
                        'estimation_std_all': np.std(estimator.itds - stim_itd),
                        'population_mean': np.mean(best_itds),
                        'delta_itds_mean': np.mean(estimator.itds[eligible] - stim_itd),
                        'delta_itds_mean_all': np.mean(estimator.itds - stim_itd),
                        'delta_itds_median': np.median(estimator.itds[eligible] - stim_itd),
                        'delta_itds_median_all': np.median(estimator.itds - stim_itd),
                        'delta_itds_p25': np.percentile(estimator.itds[eligible] - stim_itd, 25),
                        'delta_itds_p25_all': np.percentile(estimator.itds - stim_itd, 25),
                        'delta_itds_p75': np.percentile(estimator.itds[eligible] - stim_itd, 75),
                        'delta_itds_p75_all': np.percentile(estimator.itds - stim_itd, 75),
                        'itds_iqr': np.percentile(estimator.itds[eligible], 75) - np.percentile(estimator.itds[eligible], 25),
                        'itds_iqr_all': np.percentile(estimator.itds, 75) - np.percentile(estimator.itds, 25),
                        'delta_itds_iqr': np.percentile(estimator.itds[eligible] - stim_itd, 75) - np.percentile(estimator.itds[eligible] - stim_itd, 25),
                        'delta_itds_iqr_all': np.percentile(estimator.itds - stim_itd, 75) - np.percentile(estimator.itds - stim_itd, 25),
                        'hemisphere_sign': -1 if ('right' in s.meta['position'] and bool(s.meta['position']['right'])) else 1,
                    })
        self.sites_estimators = sites_estimators
    _lazy['sites_estimators'] = '_pv_decode'
    
    def filter_sites_estimators(self, site = None, bc = None, stim_itd = None):
        ses = []
        for se in self.sites_estimators:
            if site is None or se['site'] == site:
                if bc is None or se['bc'] == bc:
                    if stim_itd is None:
                        ses.append(se)
                    elif isinstance(stim_itd, (int, float)) and se['stim_itd'] == stim_itd:
                        ses.append(se)
                    else:
                        try:
                            if se['stim_itd'] in list(stim_itd):
                                ses.append(se)
                        except:
                            pass
        return ses

    def get_exemplary_site_estimator(self, bc = None, stim_itd = None):
        if stim_itd is None:
            stim_itd = self.exemplary_site.showcase_stim_itds
        """Get a list of matching sites_estimators for exemplary_site"""
        return self.filter_sites_estimators(site = self.exemplary_site,
            bc = bc, stim_itd = stim_itd)

    def _sidepeak_data(self):
        bc = 100
        sp_data = []
        for s in self.itd_sites:
            for u in s.units.values():
                resp = u.responses(('itd_bc', 0))
                resp_mean = resp.groupby(['bc', 'itd']).mean()
                resp_itd = resp_mean.loc[bc].index.get_level_values('itd').values
                resp_val = resp_mean.loc[bc, u.name].values
                try:
                    d = side_peak_relation(resp_itd, resp_val)
                    d.update({'unit': u})
                    sp_data.append(d)
                except:
                    print("Cannot find side peak response:", u, resp_itd, resp_val, sep='\n', end='\n\n')
        self.sp_data = sp_data
        self.sps_indexes = np.asarray([d['rel_Y'] for d in sp_data])
        self.sps_indexes.sort()
    _lazy['sp_data'] = '_sidepeak_data'
    _lazy['sps_indexes'] = '_sidepeak_data'


##
# PV Decoder parts
##

class PopulationProfile:
    """Representation of the ideal/model population response profile
    
    This is equal to the likelihood function from [1]_. Object is
    callable and returns (normalized) response rates for given (best)
    ITDs.
    
    Note: Binaural correlation (bc) varies from 0% to 100%, whereas the
    model used interaural correlation (IC) from 0 to 1. See formula
    below Fig 2 in [1]_.
    
    References
    ----------
    .. [1]: Fischer BJ, Peña JL (2011) Owl’s behavior and neural
       representation predicted by Bayesian inference.
       Nat Neurosci 14:1061–1066, https://doi.org/10.1038/nn.2872
    """

    # width of likelihood from
    _w_const = 41.2

    def __init__(self, stim_itd = 0, bc = None, fixed_w = None):
        """Initialize PopulationProfile"""
        self.stim_itd = stim_itd
        if fixed_w is not None:
            self.w = fixed_w
        else:
            self.bc = bc
    
    def __setattr__(self, name, value):
        """Ensure self.w is recalculated whenever self.bc is set."""
        object.__setattr__(self, name, value)
        if name == 'bc':
            self._update_width()
    
    def _update_width(self):
        """Update width `self.w` of the likelihood function
        
        Will be called automatically whenever self.bc is set, to make
        the object consistent.
        """
        if self.bc is None:
            self.w = self._w_const
        else:
            # Width including term for binaural correlation (BC)
            self.w = (self._w_const +
                      219.34 * np.exp(-.1131 * float(self.bc))
                     )
    
    def __call__(self, itd):
        """Calculate PopulationProfile at (units' best) ITD(s)"""
        return np.exp(
            -0.5 * ((np.asarray(itd) - self.stim_itd) / self.w) ** 2
        )


def optimal_scale(func, x, y):
    """Find the optimal scale factor `c` for function `func`, minimizing
    the error of:
        y = c * func(x)
    where x and y are equally sized arrays
    """
    y_unscaled = func(x)
    return np.dot(y_unscaled, y) / np.dot(y_unscaled, y_unscaled)


class PVdecoder(metaclass=LazyLoader):
    """Estimator of stimulus ITD based on PV readout model.
    
    
    """

    def __init__(self, xvalues, yvalues, stim_itd,
            bc = None, fixed_w = None,
            scale = None, itd_bounds = 200.):
        """"""
        self.x = np.asarray(xvalues).squeeze()
        self.y = np.asarray(yvalues).squeeze()
        self.stim_itd = stim_itd
        self.bc = bc
        self.fixed_w = fixed_w
        if scale is not None:
            # Force use given scale, calculated in _scale() otherwise:
            self.scale = scale
        try:
            # For individual bounds as sequence:
            if len(bounds) == 2:
                self.estimation_bounds = [tuple(itd_bounds)]
        except:
            # For numeric bounds:
            self.estimation_bounds = [
                (self.stim_itd - itd_bounds, self.stim_itd + itd_bounds)
            ]
        ### SPLIT HERE
        # Fitted estimation function (shifted, not scaled):
        self.f_fit = PopulationProfile(stim_itd=self.itd, bc = self.bc, fixed_w = self.fixed_w)
        # Optimized estimation function (scaled and shifted):
        self.f_opt = lambda itd: self.scale * self.f_fit(itd)
        
    def _y_mean(self):
        """Mean response (potentially across trials)"""
        if self.y.ndim > 1:
            self.y_mean = np.mean(self.y, axis=0)
        else:
            self.y_mean = self.y
    _lazy['y_mean'] = '_y_mean'
    
    def _scale(self):
        """Optimized scale to minimize errors between f_norm and y_mean"""
        self.scale = optimal_scale(self.f_norm, self.x, self.y_mean)
    _lazy['scale'] = '_scale'
    
    def _f_norm(self):
        """Normalized estimation function"""
        self.f_norm = PopulationProfile(stim_itd=self.stim_itd, bc = self.bc, fixed_w = self.fixed_w)
    _lazy['f_norm'] = '_f_norm'
    
    def f_mod(self, itd):
        """Scaled version of f_norm"""
        return self.scale * self.f_norm(itd)
        
    def _eval_mean(self):
        """The actual decoding work"""
        error_fun = self.make_error_function(self.y_mean)
        self.minimize_result = sp.optimize.minimize(error_fun, (self.stim_itd, ))
        self.itd = self.minimize_result.x[0]
    _lazy['minimize_result'] = '_eval_mean'
    _lazy['itd'] = '_eval_mean'

    def _eval_per_trial(self):
        """"""
        
        if not self.y.ndim > 1:
            # No individual trials
            self.minimize_results = [self.minimize_result]
            self.itds = np.array([self.itd])
            return
        self.minimize_results = []
        for y in self.y:
            error_fun = self.make_error_function(y)
            self.minimize_results.append(
                sp.optimize.minimize(
                    error_fun,
                    (self.stim_itd, ),
                    bounds = self.estimation_bounds
                )
            )
        self.itds = np.array([min_res.x[0] for min_res in self.minimize_results])
    _lazy['minimize_results'] = '_eval_per_trial'
    _lazy['itds'] = '_eval_per_trial'

    def make_error_function(self, myy):
        # Create PopulationProfile to be used in error function
        # The stim_itd will be *changed* during optimization, don't
        # use this object later. Initializing with actual stim_itd:
        p = PopulationProfile(stim_itd = self.stim_itd, bc = self.bc, fixed_w = self.fixed_w)
        def error_fun(x):
            """Error function to minimize with scipy.optimize.minimize"""
            # change p.stim_itd, this is what's being optimized:
            p.stim_itd = x[0]
            return np.sqrt(
                np.mean(
                    (self.scale * p(self.x) - myy) ** 2
                )
            )
        return error_fun

def normalize_stim_itds(best_itds, stim_itds, flip_left = True):
    """Given a list/array of best ITDs, normalize a list of stimulus ITDs.
    
    The returned array `normed`:
    
    * normed[k] corresponds to stim_itds[k]
    * normed[k] == -1 means stim_itds[k] == min(best_itds)
    * normed[k] == +1 means stim_itds[k] == max(best_itds)
    
    If flip_left is True (default), the sign of normed will be inverted for
    best_itds that are negative on average (recorded on the right hemisphere).
    This way, `normed[k] == -1` indicates the frontal limit of the range of
    best_itds, and `normed[k] == +1` indicates the peripheral limit.
    """
    best_itds = sorted(best_itds)
    normed = -1 + 2 * (np.asarray(stim_itds) - best_itds[0]) / abs(best_itds[-1] - best_itds[0])
    if flip_left and np.mean(best_itds) < 0:
        normed = -normed
    return normed


##
# Extension for pedman:
##

def unit_response_scale_factor(u, index, ref_index = ('itd_bc', 0), depvar='itd', by_bc = False):
    resp = u.responses(index)
    resp_ref = u.responses(ref_index)
    resp_mean = resp.groupby(['bc', depvar]).mean()
    resp_ref_mean = resp_ref.groupby(['bc', depvar]).mean()

    scale_factors = {}

    if not isinstance(by_bc, (bool)):
        bcs = [by_bc]
    else:
        bcs = resp_mean.index.unique('bc')

    for kbc, bc in enumerate(bcs):
        resp_itd = resp_mean.loc[bc].index.get_level_values('itd').values
        resp_val = resp_mean.loc[bc, u.name].values
        resp_ref_itd = resp_ref_mean.loc[bc].index.get_level_values('itd').values
        resp_ref_val = resp_ref_mean.loc[bc, u.name].values

        resp_optimal = np.interp(resp_itd, resp_ref_itd, resp_ref_val)

        scale_optimal = np.dot(resp_val, resp_optimal) / np.dot(resp_val, resp_val)

        scale_factors[bc]  = scale_optimal

    if by_bc is True:
        return scale_factors
    elif by_bc in scale_factors:
        return scale_factors[by_bc]
    else:
        return np.mean(list(scale_factors.values()))

from pedman.site import Unit
Unit.responses_scale_factor = unit_response_scale_factor
del Unit

##
# Side Peak Responses:
##

def find_side_peaks(X, Y, max_I=None):
    """
    For X|Y-data return indexes of all relative (but not the absolute) maxima
    """
    if max_I is None:
        max_I = np.argmax(Y)
    allsidepeaks_I = []
    for k in range(1, len(Y)-1):
        if k == max_I:
            # Do not include the main peak
            continue
        if Y[k-1] >= Y[k]:
            # Value is smaller than left value
            continue
        # From here on true: Y[k-1] < Y[k]
        if Y[k] > Y[k+1]:
            allsidepeaks_I.append(k)
        elif Y[k] == Y[k+1]:
            # Possible "plateau-peak":
            for l in range(k+1, len(Y)-1):
                if Y[l] == Y[l+1]:
                    # Continuation of plateau
                    continue
                elif Y[l] > Y[l+1]:
                    # End of plateau, next smaller.
                    # Return first index of equal values
                    allsidepeaks_I.append(k)
                else:
                    # Y[l] < Y[l+1]
                    # Next value higher. No local maximum
                    break
    return np.array(allsidepeaks_I)

def find_troughs(X, Y):
    '''
    For X|Y-data return indexes of all troughs
    '''
    mins_I = find_side_peaks(X, -Y, max_I=0)
    return mins_I
    #return mins_X = X[mins_I]
    #return mins_Y = Y[mins_I]

def side_peak_relation(X, Y, sp_option='greatest primary', tr_option='nearest' , **kwargs):
    """From X|Y-data calculate the Y-relation of main and highest side peak
    
    Keyword arguments:
        max_I       - Force max_I, the index of max(Y), instead of searching for
                      the maximum in Y
        side_I      - Force side_I, the index of the side peak in X|Y, instead
                      of searching it in Y
        mins_I      - Force mins_I, the indices of the troughs
        min_X_diff  - minimal difference of max_X and side_X
        sp_option   - Choose between 'greatest primary'(default): uses greatest primary side peak;
                      'greatest': uses greatest side peak; 'mean': uses mean values of the two primary side peaks
                      (tr_option must be 'minimum')
        tr_option   - Choose between 'nearest' (default): uses nearest trough to SP; 'minimum': uses absolute minimum of curve
    
    Return dict with fields:
        max_Y    - max(Y)
        max_X    - Value in X corresponding to max_Y
        side_Y   - secondary peak
        side_X   - Value in X corresponding to side_Y
        min_Y    - min(Y)
        min_X    - Value in X corresponding to min_Y
        diff_Y   - dynamic range, max_Y - min_Y
        rel_Y    - relative Y of secondary peak, where min_Y = 0 and max_Y = 1
    """
    X = X.reshape((-1))
    Y = Y.reshape((-1))
    ## Main peak data
    if 'max_I' in kwargs:
        max_I = kwargs['max_I']
    else:
        max_I = np.argmax(Y)
    max_X = X[max_I]
    max_Y = Y[max_I]
    ## Side peak data
    if 'side_I' in kwargs:
        side_I = kwargs['side_I']
    else:
        sides_I = find_side_peaks(X, Y, max_I=max_I)
        if 'min_X_diff' in kwargs:
            sides_I = sides_I[np.abs(X[sides_I]-max_X) >= kwargs['min_X_diff']]
        if sp_option == 'greatest primary':
            # Use the greatest primary side peak of the derived side peaks
            primaries_I = []
            if np.any((max_I - sides_I) > 0):
                primaries_I += [np.max(sides_I[(max_I - sides_I) > 0])]
            if np.any((max_I - sides_I) < 0):
                primaries_I += [np.min(sides_I[(max_I - sides_I) < 0])]
            #primaries_I = np.array(primaries_I)
            side_I = primaries_I[np.argmax(Y[primaries_I])]
            #side_L = np.max(sides_I[(max_I - sides_I) > 0]) # closest prim. SP to MP from left side
            #side_R = np.min(sides_I[(max_I - sides_I) < 0]) # closest prim. SP to MP from rightside
            #side_I = side_L if Y[side_L] > Y[side_R] else side_R # use greatest prim. SP
            side_X = X[side_I]
            side_Y = Y[side_I]
        if sp_option == 'greatest':
            # Use greatest side peak
            try:
                side_I = sides_I[np.argmax(Y[sides_I])]
            except IndexError as e:
                print(sides_I)
                raise e
            side_X = X[side_I]
            side_Y = Y[side_I]
        if sp_option == 'mean':
            # Use mean value of the two primary side peaks
            side_L_I = max(sides_I[(max_I - sides_I) > 0]) # closest prim. SP to MP from left side
            side_R_I = min(sides_I[(max_I - sides_I) < 0]) # closest prim. SP to MP from rightside
            side_L_X = X[side_L_I]
            side_L_Y = Y[side_L_I]
            side_R_X = X[side_R_I]
            side_R_Y = Y[side_R_I]
            side_Y = np.mean([side_L_Y, side_R_Y])
    ## Minimum data (all troughs)
    if 'mins_I' in kwargs:
        mins_I = kwargs['mins_I']
    else:
        mins_I = find_troughs(X, Y)
    #mins_X = X[mins_I]
    #mins_Y = Y[mins_I]
    # Minimum data
    if tr_option == 'nearest':
        if sp_option == 'mean':
            min_I = np.argmin(Y) # otherwise no value for min_I
        else:
            if side_I < max_I:
                min_I = min(mins_I[(side_I - mins_I) < 0])
            else: # side_I can't be max_I (side_I == max_I does not occur)
                min_I = max(mins_I[(side_I - mins_I) > 0])
    if tr_option == 'minimum':
            min_I = np.argmin(Y)
    min_X = X[min_I]
    min_Y = Y[min_I]
    # Dynamic range
    diff_Y = max_Y - min_Y
    # Relative secondary peak height
    rel_Y = (side_Y - min_Y) / diff_Y
    if sp_option == 'mean':
        return dict(max_X = max_X, max_Y = max_Y, min_X = min_X, min_Y = min_Y,
                side_X = [side_L_X, side_R_X], side_Y = side_Y,
                diff_Y = diff_Y, rel_Y = rel_Y,
                )
    else:
        return dict(max_X = max_X, max_Y = max_Y, min_X = min_X, min_Y = min_Y,
                side_X = side_X, side_Y = side_Y,
                diff_Y = diff_Y, rel_Y = rel_Y,
                )
