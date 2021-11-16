"""Internal base and metaclasses for pedman"""

class LazyLoader(type):
    """Metaclass handling lazy loading or calculation of attributes
    
    This metaclass standardizes the handling of lazy (on demand) loaded
    attributes by creating a `_lazy` dict in which each key is the name
    of a lazy attribute and the value is the name of the method that
    sets this attribute.
    
    The methods referenced in the `_lazy` may not depend on any
    arguments beside `self`. Methods that do have additional parameters,
    optional or mandatory, are not suitable for lazy loading.
    
    Examples
    --------
    This examples show a class with two lazy attributes `foo` and `bar`,
    which will be computed when first one of them is accessed.
    
    class RandomPair(metaclass=LazyLoader):
        '''A class with to lazy random numbers.'''
        def roll_dice(self):
            self.foo = np.random.rand()
            self.bar = np.random.rand()
        _lazy['foo'] = 'roll_dice'
        _lazy['bar'] = 'roll_dice'
    """
    
    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        d = dict(_lazy = {})
        for b in bases:
            if hasattr(b, '_lazy'):
                d['_lazy'].update(b._lazy)
        d['__getattr__'] = LazyLoader._getattr
        d['__dir__'] = LazyLoader._dir
        d['islazy'] = LazyLoader._islazy
        d['islazyloaded'] = LazyLoader._islazyloaded
        d['lazyunload'] = LazyLoader._lazyunload
        d['lazydir'] = LazyLoader._lazydir
        return d

    @staticmethod
    def _getattr(self, name):
        if name in self._lazy:
            # Find the function needed to make attribute `name`
            # available, and execute it:
            getattr(self, self._lazy[name])()
            # Make sure the attribute is now in the objects __dict__:
            if not name in self.__dict__:
                raise AttributeError(
                    ("Executing `{cls}.{fun}()` did not make `{att}` "
                     "available as promised by `_lazy` entry.").format(
                        cls = self.__class__.__name__,
                        fun = self._lazy[name],
                        att = name,
                    )
                )
            # Return the (now available) attribute:
            return getattr(self, name)
        # Raise AttributeError, as expected from __getattr__ methods:
        raise AttributeError(
            "{cls} has not attribute '{att}'".format(
                cls = self.__class__.__name__,
                att = name,
            )
        )

    @staticmethod
    def _dir(self):
        """__dir__ method for LazyLoader classes"""
        d = []
        for b in self.__class__.__bases__:
            d.extend(object.__dir__(self))
        d.extend(self._lazy.keys())
        return list(set(d))
        
    def _islazy(self, name):
        """Checks if an `name` is a lazy loaded attribute.
        
        Returns
        -------
        bool
            `True` if the name belongs to a lazy attribute.
        """
        return name in self._lazy
    
    @staticmethod
    def _islazyloaded(self, name):
        """Check if a lazy attribute was already loaded.
        
        Returns
        -------
        bool or None
            `True` if a lazy attribute is loaded, `False` otherwise.
            `None` is returned if the attribute is not lazy.
        """
        if self.islazy(name):
            return name in self.__dict__
        else:
            return None
    
    @staticmethod
    def _lazyunload(self, name):
        """Unloads or deletes a lazy loaded attribute."""
        if self.islazyloaded(name):
            delattr(self, name)
    
    @staticmethod
    def _lazydir(self):
        """Get a directory of lazy attributes and their load status
        
        Returns
        -------
        dict
            Keys are the attribute names and values indicate if this
            attribute is currently loaded.
        """
        return {n: self.islazyloaded(n) for n in self._lazy.keys()}
