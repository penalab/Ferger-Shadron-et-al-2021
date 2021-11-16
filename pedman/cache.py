"""Extended cache module for pedman

This module introduces a caching mechanism to the most expensive pedman
functions by "wrapping" them with cache aware functions from this module.

**Experimental: This cache method is still in development. Use with caution!**
Especially, there are no sanity checks, yet. Data loaded from the cache might
_not_ be in agreement with the actual data or meta files!

Usage
-----

Caching has to be enabled/initialized before it has an effect:

```
import pedman.cache
pedman.cache.PedmanCache()
```

A single PedmanCache instance will be created and handle file based caching.

"""

import os
import functools
import pandas as pd
import numpy as np
from . import site

class PedmanCache():
    """Pedman Cache Class - File based cache
    
    A single PedmanCache instance will be created and handle file based caching.
    """
    
    _instance = None
    _origs = {}
    
    def __new__(cls, CACHE_DIR = "~/.pedman/cache", options = {}):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("Enabled pedman.cache")
        cls._instance.CACHE_DIR = os.path.abspath(os.path.expanduser(CACHE_DIR))
        cls._instance.wrap_trialparams(options.get('trialparams', True))
        cls._instance.wrap_responses(options.get('responses', True))
        return cls._instance
    
    def __del__(self):
        if self is self.__class__._instance:
            self.__class__.disable()
    
    @classmethod
    def disable(cls):
        print("Disabled pedman.cache")
        if not PedmanCache._instance is None:
            PedmanCache._instance.wrap_trialparams(wrap = False)
            PedmanCache._instance.wrap_responses(wrap = False)
        PedmanCache._instance = None
        
    def site_cache_dir(pedcache, s, create = True):
        d = os.path.join(
            pedcache.CACHE_DIR,
            f"{s.id()}"
        )
        if create and not os.path.exists(d):
            os.mkdir(d)
        return d
    
    def wrap_trialparams(pedcache, wrap = True):
        if not 'trialparams' in pedcache._origs:
            # Remember unwrapped original:
            pedcache._origs['trialparams'] = site.Recording.trialparams
        if wrap:
            if site.Recording.trialparams is pedcache._origs['trialparams']:
                @functools.wraps(pedcache._origs['trialparams'])
                def trialparams_wrapper(self):
                    cache_fn = f"rec{self.index}--trialparams.npz"
                    cache_path = os.path.join(pedcache.site_cache_dir(self.site), cache_fn)
                    if os.path.exists(cache_path):
                        try:
                            with np.load(cache_path, allow_pickle=True) as l:
                                conditions = [
                                    'xdphys_hash' in l and self._meta['xdphys_hash'] == l['xdphys_hash'],
                                ]
                                if all(conditions):
                                    trialparams = pd.DataFrame(
                                        {k: l[k] for k in l.files if not k in ('xdphys_hash', )}
                                    ).set_index('index')
                                    return trialparams
                        except:
                            os.remove(cache_path)
                            raise RuntimeError(f"Could not load cache from {cache_path}")
                    # print(f"Recollecting trialparams for {self}")
                    trialparams = pedcache._origs['trialparams'](self)
                    df = trialparams.reset_index()
                    np.savez(cache_path,
                        xdphys_hash = self._meta['xdphys_hash'],
                        **{c: df[c] for c in df.columns}
                    )
                    return trialparams
                site.Recording.trialparams = trialparams_wrapper
        else:
            # Restore original
            site.Recording.trialparams = pedcache._origs['trialparams']
    
    def wrap_responses(pedcache, wrap = True):
        if not 'responses' in pedcache._origs:
            # Remember unwrapped original:
            pedcache._origs['responses'] = site.Recording.responses
        if wrap:
            if site.Recording.responses is pedcache._origs['responses']:
                @functools.wraps(pedcache._origs['responses'])
                def responses_wrapper(self, spikes_per_second=False):
                    cache_fn = f"rec{self.index}--responses.npz"
                    cache_path = os.path.join(pedcache.site_cache_dir(self.site), cache_fn)
                    if os.path.exists(cache_path):
                        try:
                            with np.load(cache_path, allow_pickle=True) as l:
                                conditions = [
                                    'stimtime' in l and self.stimtime() == tuple(l['stimtime']),
                                    'units' in l and tuple(self.site.units.keys()) == tuple(l['units']),
                                    'plexon_hash' in l and self._meta['plexon_hash'] == l['plexon_hash'],
                                ]
                                if all(conditions):
                                    responses = self.trialparams()
                                    responses.set_index(list(responses.columns), append=True, inplace=True)
                                    responses = responses.join(
                                        pd.DataFrame(l['responses'], columns = l['units']),
                                        on='index'
                                    )
                                    if spikes_per_second:
                                        responses = responses / (l['stimtime'][1] - l['stimtime'][0])
                                    return responses
                        except:
                            os.remove(cache_path)
                            raise RuntimeError(f"Could not load cache from {cache_path}")
                    # print(f"Recollecting responses for {self}")
                    responses = pedcache._origs['responses'](self, spikes_per_second=False)
                    np.savez(cache_path,
                        responses = responses.values,
                        units = responses.columns.values,
                        stimtime = self.stimtime(),
                        plexon_hash = self._meta['plexon_hash'],
                    )
                    if spikes_per_second:
                        stimstart, stimend = self.stimtime()
                        responses = responses / (stimend - stimstart)
                    return responses
                site.Recording.responses = responses_wrapper
        else:
            # Restore original
            site.Recording.responses = pedcache._origs['responses']
