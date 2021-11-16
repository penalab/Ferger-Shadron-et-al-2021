

import datetime
from collections import namedtuple
from . import pypl2

def pl2_spikeinfo(filename, channel):
    """Largely adopted from pl2_spikes
    """
    #Create an instance of PyPL2FileReader.
    p = pypl2.PyPL2FileReader()
    
    #Verify that the file passed exists first.
    #Open the file.
    handle = p.pl2_open_file(filename)

    #If the handle is 0, print error message and return 0.
    if (handle == 0):
        print_error(p)
        return 0
        
    #Create instance of PL2FileInfo.
    file_info = pypl2.PL2FileInfo()
    
    res = p.pl2_get_file_info(handle, file_info)
    
    #If res is 0, print error message and return 0.
    if (res == 0):
        print_error(p)
        return 0        
    
    #Create instance of PL2SpikeChannelInfo.
    schannel_info = pypl2.PL2SpikeChannelInfo()
    
    #Check if channel is an integer or string, and call appropriate function
    if type(channel) is int:
        res = p.pl2_get_spike_channel_info(handle, channel, schannel_info)
    if type(channel) is str:
        res = p.pl2_get_spike_channel_info_by_name(handle, channel, schannel_info)
    
    # Close the file
    p.pl2_close_file(handle)
    
    #If res is 0, print error message and return 0.
    if (res == 0):
        print_error(p)
        return 0    

    PL2SpikeInfo = namedtuple('PL2SpikeInfo', 'adfrequency samples pre_samples')
    
    return PL2SpikeInfo(schannel_info.m_SamplesPerSecond,
                        schannel_info.m_SamplesPerSpike,
                        schannel_info.m_PreThresholdSamples)

def pl2_fileinfo(filename):
    p = pypl2.PyPL2FileReader()

    handle = p.pl2_open_file(filename)

    if (handle == 0):
        pypl2.print_error(p)
        raise IOError("Error occured. Cannot get file handle.")

    #Create instance of PL2FileInfo.
    file_info = pypl2.PL2FileInfo()

    res = p.pl2_get_file_info(handle, file_info)

    index_event_channel = 8
    
    echannel_info = pypl2.PL2DigitalChannelInfo()
    res = p.pl2_get_digital_channel_info(handle, index_event_channel, echannel_info)
    
    n = echannel_info.m_NumberOfEvents
    
    t = file_info.m_CreatorDateTime

    # Close the file
    p.pl2_close_file(handle)

    PL2FileInfo = namedtuple('PL2FileInfo', 'datetime n_trials')
    
    return PL2FileInfo(
        datetime.datetime(
            year = t.tm_year + 1900, # year 2018 is represented as 118
            month = t.tm_mon + 1, # month 9 (September) is represented as 8
            day = t.tm_mday,
            hour = t.tm_hour,
            minute = t.tm_min,
            second = t.tm_sec,
            microsecond = 1000 * file_info.m_CreatorDateTimeMilliseconds
        ),
        n
    )

# Link into original pypl2 package:
pypl2.pl2_spikeinfo = pl2_spikeinfo
pypl2.pl2_fileinfo = pl2_fileinfo
