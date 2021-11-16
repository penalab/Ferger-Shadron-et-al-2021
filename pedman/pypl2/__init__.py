# __init__.py - Module setup for PyPL2
#
# (c) 2016 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

#__init__.py serves three purposes:
#   1) Let's Python know that the .py files in this directory are importable modules.
#   2) Sets up classes and functions in the pypl2lib and pypl2api modules to be easy to
#      access. For example, without importing the pl2_ad function from pypl2api in 
#      __init__.py, you would have to import pypl2 in your script like this:
#           from pypl2.pypl2api import pl2_ad
#      instead of like this:
#           from pypl2 import pl2_ad
#      It's a minor convenience, but improves readability.
#   3) Explicitly states which classes and functions in PyPL2 are meant to be public 
#      parts of the API.

from .pypl2lib import PL2FileInfo, PL2AnalogChannelInfo, PL2SpikeChannelInfo, PL2DigitalChannelInfo, PyPL2FileReader
from .pypl2api import pl2_ad, pl2_spikes, pl2_events, pl2_info

__author__ = 'Chris Heydrick (chris@plexon.com)'
__version__ = '1.1.0'

# 5/24/2016 CH
# Added 64-bit .dll support, incremented to 1.1.0