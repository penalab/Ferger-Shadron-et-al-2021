"""Handle meta files

Site-YAML files (.yaml) are meant to ease the organization and use of ephys data
from different sources. They:

* list corresponding files from xdphys and Plexon
* Adding some descriptive (meta-)data
"""

import sys
import datetime
import numpy as np

try:
    import ruamel.yaml
except ImportError:
    print("""Couldn't import ruamel.yaml. For installation instructions,
    see: https://yaml.readthedocs.io/en/latest/install.html""", file=sys.stderr)
    raise


def simplify(obj, dict_cls = dict, list_cls = list, other_cls = None):
    """Helper function to simplify nested structures
    
    Recursively casts nested structures into builtin types, with the following
    rules applied in order:
    
    - int, float, complex, list, tuple, str and None remain unchanged
    - dict-subclasses are turned into dict_cls objects (default: dict)
    - list-, tuple-, and set-subclasses are turned into list_cls objects
      (default: list)
    - datetime.date, .datetime and .time objects are turned into str objects
    - other objects are turned into other_cls objects if the argument is given,
      otherwise a ValueError will be raised

    dict_cls, list_cls and other_cls arguments may also be functions returning
    objects of the intended class.
    Objects returned by dict_cls must support the dict interface - most notably
    they must implement items(), keys() and key-indexing.
    Objects returned by other_cls are NOT further recursively simplified. If
    that is desired, it must be done by the other_cls-function.
    """
    kwargs = {
        'dict_cls': dict_cls,
        'list_cls': list_cls,
        'other_cls': other_cls,
    }
    if isinstance(obj, (int, float, complex, str)) or obj is None:
            return obj
    elif isinstance(obj, dict):
        dobj = dict_cls(obj.items())
        for k in dobj.keys():
            dobj[k] = simplify(dobj[k], **kwargs)
        return dobj
    elif isinstance(obj, (list, tuple, set)):
        return list_cls(simplify(e, **kwargs) for e in obj)
    elif isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    elif other_cls is not None:
        return other_cls(obj)
    else:
        raise ValueError(f"I don't know how to simplify type {type(obj)}, value: {obj}")


def load(filename):
    with open(filename, "r") as f:
        return ruamel.yaml.round_trip_load(f)


def dump(data, filename, druginfo_anchors = True):
    if druginfo_anchors:
        if 'log' in data and '_druginfo' in data['log']:
            for k, v in data['log']['_druginfo'].items():
                if not isinstance(v, ruamel.yaml.comments.CommentedMap):
                    data['log']['_druginfo'][k] = ruamel.yaml.comments.CommentedMap(v)
                    v = data['log']['_druginfo'][k]
                if v.yaml_anchor() is None:
                    v.yaml_set_anchor(k)
                v.anchor.always_dump = True
    with open(filename, 'w', newline='\n') as f:
        ruamel.yaml.round_trip_dump(data, stream = f, indent=4, default_flow_style=False)
