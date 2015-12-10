# -*- coding: utf-8 -*-
"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV and NSE file formats.

Depends on: numpy

Supported: Read

Author: jsprenger,ccanova
Adapted from the exampleIO of python-neo
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division


import sys
import os
import warnings
import copy
import re
import datetime
import pkg_resources

import numpy as np
if pkg_resources.pkg_resources.parse_version(np.__version__) < pkg_resources.pkg_resources.parse_version('1.9.2'):
     raise ImportError("Using numpy version %s. Version must be >= 1.9.2" % (np.__version__))

import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import (Block, Segment,
                      RecordingChannel, RecordingChannelGroup, AnalogSignalArray,
                      SpikeTrain, EventArray,Unit)
from os import listdir, sep
from os.path import isfile, getsize

import hashlib
import pickle

class NeuralynxIO(BaseIO):
    """
    Class for reading Neuralynx files.

    It enables reading:
    - :class:'Block'
    - :class:'Segment'
    - :class:'AnalogSignalArray'
    - :class:'SpikeTrain'

    Usage:
        from neo import io
        import quantities as pq
        import matplotlib.pyplot as plt

        session_folder = '../Data/2014-07-24_10-31-02'
        NIO = io.NeuralynxIO(session_folder,print_diagnostic = True)
        block = NIO.read_block(t_starts = 0.1*pq.s, t_stops = 0.2*pq.s,events=True)
        seg = block.segments[0]
        analogsignal = seg.analogsignalarrays[0]
        plt.plot(analogsignal.times.rescale(pq.ms), analogsignal.magnitude)
        plt.show()

    """


    is_readable = True  # This class can only read data
    is_writable = False  # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects = [Segment, AnalogSignalArray, SpikeTrain, EventArray]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects = [Segment, AnalogSignalArray, SpikeTrain]
    # This class is not able to write objects
    writeable_objects = []


    has_header = False
    is_streameable = False

    # This is for GUI stuff : a definition for parameters when reading.
    # This dict should be keyed by object (`Block`). Each entry is a list
    # of tuple. The first entry in each tuple is the parameter name. The
    # second entry is a dict with keys 'value' (for default value),
    # and 'label' (for a descriptive name).
    # Note that if the highest-level object requires parameters,
    # common_io_test will be skipped.
    read_params = {
        Segment: [('waveforms', {'value':True})],
        Block: [('waveforms', {'value': False})]
        }

    # do not supported write so no GUI stuff
    write_params = None

    name = 'Neuralynx'
    description = 'This IO reads .nse/.ncs/.nev files of the Neuralynx (Cheetah) recordings system (tetrodes).'

    extensions = ['nse', 'ncs', 'nev', 'ntt']

    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or
    # a database this info is for GUI stuff also
    mode = 'dir'

    # hardcoded parameters from manual, which are not present in Neuralynx data files
    # unit of timestamps in different files
    nev_time_unit = pq.microsecond
    ncs_time_unit = pq.microsecond
    nse_time_unit = pq.microsecond
    ntt_time_unit = pq.microsecond
    # unit of sampling rate in different files
    ncs_sr_unit = pq.Hz
    nse_sr_unit = pq.Hz
    ntt_sr_unit = pq.Hz



    def __init__(self, sessiondir=None, cachedir = None, use_cache='hash', print_diagnostic=False, filename=None):
        """
        Arguments:
            sessiondir: the directory the files of the recording session are
                            collected. Default 'None'.
            print_diagnostic: indicates, whether information about the loading of
                            data is printed in terminal or not. Default 'False'.
            cachedir: the directory where metadata about the recording session is
                            read from and written to.
            use_cache: method used for cache identification. Possible values: 'hash'/
                            'always'/'datesize'/'never'. Default 'hash'
            filename: this argument is handles the same as sessiondir and is only
                            added for external IO interfaces. The value of sessiondir
                            has priority over filename.
        """

        BaseIO.__init__(self)

        # possiblity to provide filename instead of sessiondir for IO compatibility
        if filename != None and sessiondir==None:
            sessiondir = filename

        if sessiondir == None:
            raise ValueError('Must provide a directory containing data files of'
                                ' of one recording session.')

        # remove filename if specific file was passed
        if any([sessiondir.endswith('.%s'%ext) for ext in self.extensions]):
            sessiondir = sessiondir[:sessiondir.rfind(sep)]

        # remove / for consistent directory handling
        if sessiondir.endswith(sep):
            sessiondir = sessiondir.rstrip(sep)

        # set general parameters of this IO
        self.sessiondir = sessiondir
        self.filename = sessiondir.split(sep)[-1]
        self._print_diagnostic = print_diagnostic
        self.associated = False
        self._associate(cachedir=cachedir,usecache=use_cache)

        self._diagnostic_print('Initialized IO for session %s'%self.sessiondir)



    def read_block(self, lazy=False, cascade=True, t_starts=[None], t_stops=[None],
                    electrode_list=[], units=[], analogsignals=True, events=False,
                    waveforms = False):
        """
        Reads data in a requested time window and returns block with as many segments
        es necessary containing these data.

        Arguments:
            lazy : Postpone actual reading of the data files. Default 'False'.
            cascade : Do not postpone reading subsequent neo types (segments).
                            Default 'True'.
            t_starts : list of quantities or quantity describing the start of the
                            requested time window to load. If None or [None]
                            the complete session is loaded. Default '[None]'.
            t_stops : list of quantities or quantity describing the end of the
                            requested time window to load. Has to contain the
                            same number of values as t_starts. If None or [None]
                            the complete session is loaded. Default '[None]'.
            electrode_list : list of integers containing the IDs of the requested
                            to load. If [] all available channels will be loaded.
                            Default: [].
            units : list of integers containing the IDs of the requested units
                            to load. If [] all available units will be loaded.
                            Default: [].
            analogsignals : boolean, indication whether analogsignals should be
                            read. Default: True.
            events : Loading events. If True all available events in the given
                            time window will be read. Default: False.
            waveforms : Load waveform for spikes in the requested time
                            window. Default: False.

        Returns: Block object containing the requested data in neo structures.

        Usage:
            from neo import io
            import quantities as pq
            import matplotlib.pyplot as plt

            session_folder = '../Data/2014-07-24_10-31-02'
            NIO = io.NeuralynxIO(session_folder,print_diagnostic = True)
            block = NIO.read_block(lazy = False, cascade = True,
                                   t_starts = 0.1*pq.s, t_stops = 0.2*pq.s,
                                   electrode_list = [1,5,10], units = [1,2,3],
                                   events = True, waveforms = True)
            plt.plot(block.segments[0].analogsignalarrays[0])
            plt.show()
        """
        # Create block
        bl = Block(file_origin=self.sessiondir)
        bl.name = self.filename
        if not cascade:
            return bl

        # Checking input of t_start and t_stop
        # For lazy users that specify x,x instead of [x],[x] for t_starts,t_stops
        if t_starts == None:
            t_starts = [None]
        elif type(t_starts) == pq.Quantity:
            t_starts = [t_starts]
        elif type(t_starts) != list or any([(type(i) != pq.Quantity and i != None) for i in t_starts]):
            raise ValueError('Invalid specification of t_starts.')
        if t_stops == None:
            t_stops = [None]
        elif type(t_stops) == pq.Quantity:
            t_stops = [t_stops]
        elif type(t_stops) != list or any([(type(i) != pq.Quantity and i != None) for i in t_stops]):
            raise ValueError('Invalid specification of t_stops.')

        # adapting t_starts and t_stops to known gap times (extracted in association process / initialization)
        for gap in self.parameters_global['gaps']:
            # gap=gap_list[0]
            for e in range(len(t_starts)):
                t1,t2 = t_starts[e], t_stops[e]
                gap_start = gap[1]*self.ncs_time_unit - self.parameters_global['t_start']
                gap_stop =  gap[2]*self.ncs_time_unit- self.parameters_global['t_start']
                if ((t1==t2==None)
                        or (t1==None and t2!=None and t2.rescale(self.ncs_time_unit)>gap_stop)
                        or (t2==None and t1!=None and t1.rescale(self.ncs_time_unit)<gap_stop)
                        or (t1!=None and t2!=None and t1.rescale(self.ncs_time_unit)<gap_start
                                                  and t2.rescale(self.ncs_time_unit)>gap_stop)):
                    #adapting first time segment
                    t_stops[e]=gap_start
                    #inserting second time segment
                    t_starts.insert(e+1,gap_stop)
                    t_stops.insert(e+1,t2)
                    warnings.warn('Substituted t_starts and t_stops in order to skip gap in recording session.')


        #loading all channels if empty electrode_list
        if electrode_list == []:
            electrode_list = self.parameters_ncs.keys()

        # adding a segment for each t_start, t_stop pair
        for t_start,t_stop in zip(t_starts,t_stops):
            seg = self.read_segment(lazy=lazy, cascade=cascade,
                                    t_start=t_start, t_stop=t_stop,
                                    electrode_list=electrode_list, units=units,
                                    analogsignals=analogsignals, events=events,
                                    waveforms=waveforms)
            bl.segments.append(seg)
        populate_RecordingChannel(bl, remove_from_annotation=False)

        # This create rc and RCG for attaching Units
        rcg0 = bl.recordingchannelgroups[0]
        def find_rc(chan):
            for rc in rcg0.recordingchannels:
                if rc.index==chan:
                    return rc
        for seg in bl.segments:
            for st in seg.spiketrains:
                chan = st.annotations['electrode_id']
                rc = find_rc(chan)
                if rc is None:
                    rc = RecordingChannel(index = chan)
                    rcg0.recordingchannels.append(rc)
                    rc.recordingchannelgroups.append(rcg0)
                if len(rc.recordingchannelgroups) == 1:
                    rcg = RecordingChannelGroup(name = 'Group {}'.format(chan))
                    rcg.recordingchannels.append(rc)
                    rc.recordingchannelgroups.append(rcg)
                    bl.recordingchannelgroups.append(rcg)
                else:
                    rcg = rc.recordingchannelgroups[1]
                unit = Unit(name = st.name)
                rcg.units.append(unit)
                unit.spiketrains.append(st)
            bl.create_many_to_one_relationship()

        # Adding global parameters to block annotation
        bl.annotations.update(self.parameters_global)

        return bl


    def read_segment(self,lazy=False, cascade=True, t_start=None, t_stop=None,
                        electrode_list=[], units=[], analogsignals=True,
                        events=False, waveforms=False):
        """Reads one Segment.

        The Segment will contain one AnalogSignalArray for each channel
        and will go from t_start to t_stop.

        Arguments:


            lazy : Postpone actual reading of the data files. Default 'False'.
            cascade : Do not postpone reading subsequent neo types (SpikeTrains,
                            AnalogSignalArrays, Events).
                            Default 'True'.
            t_start : time (quantity) that the Segment begins. Default None.
            t_stop : time (quantity) that the Segment ends. Default None.
            electrode_list : list of integers containing the IDs of the requested
                            to load. If [] all available channels will be loaded.
                            Default: [].
            units : list of integers containing the IDs of the requested units
                            to load. If [] all available units will be loaded.
                            Default: [].
            analogsignals : boolean, indication whether analogsignals should be
                            read. Default: True.
            events : Loading events. If True all available events in the given
                            time window will be read. Default: False.
            waveforms : Load waveform for spikes in the requested time
                            window. Default: False.


        Returns:
            Segment object containing neo objects, which contain the data.
        """

        # input check
        #loading all channels if empty electrode_list
        if electrode_list == []:
            electrode_list = self.parameters_ncs.keys()
        elif electrode_list == None:
            raise ValueError('Electrode_list can not be None.')
        elif [v for v in electrode_list if v in self.parameters_ncs.keys()]== []:
            # warn if non of the requested channels are present in this session
            warnings.warn('Requested channels %s are not present in session '
                 '(contains only %s)'%(electrode_list,self.parameters_ncs.keys()))
            electrode_list = []


        seg = Segment(file_origin=self.filename)
        if not cascade:
            return seg

        # Reading NCS Files #
        # selecting ncs files to load based on electrode_list requested
        if analogsignals:
            for chid in electrode_list:
                if chid in self.parameters_ncs:
                    file_ncs = self.parameters_ncs[chid]['filename']
                    self.read_ncs(file_ncs, seg, lazy, cascade, t_start=t_start, t_stop = t_stop)
                else:
                    self._diagnostic_print('Can not load ncs of channel %i. '
                                           'No corresponding ncs file present.'%(chid))

        # Reading NEV Files (Events)#
        # reading all files available
        if events:
            for filename_nev in self.nev_asso:
                self.read_nev(filename_nev, seg, lazy, cascade, t_start = t_start, t_stop = t_stop)

        # Reading Spike Data only if requested
        if units != None:
            # Reading NSE Files (Spikes)#
            # selecting nse files to load based on electrode_list requested
            for chid in electrode_list:
                if chid in self.parameters_nse:
                    filename_nse = self.parameters_nse[chid]['filename']
                    self.read_nse(filename_nse, seg, lazy, cascade, t_start = t_start, t_stop = t_stop, waveforms = waveforms)
                else:
                    self._diagnostic_print('Can not load nse of channel %i. '
                                           'No corresponding nse file present.'%(chid))

            # Reading ntt Files (Spikes)#
            # selecting ntt files to load based on electrode_list requested
            for chid in electrode_list:
                if chid in self.parameters_ntt:
                    filename_ntt = self.parameters_ntt[chid]['filename']
                    self.read_ntt(filename_ntt, seg, lazy, cascade, t_start = t_start, t_stop = t_stop, waveforms = waveforms)
                else:
                    self._diagnostic_print('Can not load ntt of channel %i. '
                                           'No corresponding ntt file present.'%(chid))

        return seg



    # TODO: Option to load ncs based on channel_id instead of filename? Option to load ncs without providing segment?
    def read_ncs(self, filename_ncs, seg, lazy=False, cascade=True, t_start = None, t_stop = None):
        '''
        Reading a single .ncs file from the associated Neuralynx recording session.
        In case of a recording gap between t_start and t_stop, data are only loaded until gap start.
        For loading data across recording gaps use read_block(...).

        Arguments:
            filename_ncs : Name of the .ncs file to be loaded.
            seg : Neo Segment, to which the AnalogSignalArray containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            AnalogSignalArray. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time or sample (quantity or integer) that the AnalogSignalArray begins.
                            Default None.
            t_stop : time or sample (quantity or integer) that the AnalogSignalArray ends.
                            Default None.

        Returns:
            None
        '''

        # checking format of filename and correcting if necessary
        if filename_ncs[-4:] != '.ncs':
            filename_ncs = filename_ncs + '.ncs'
        if sep in filename_ncs:
            filename_ncs = filename_ncs.split(sep)[-1]


        # Extracting the channel id from prescan (association) of ncs files with
        # this recording session
        chid = self.get_channel_id_by_file_name(filename_ncs)
        if chid == None:
            raise ValueError('NeuralynxIO is attempting to read a file '
                            'not associated to this session (%s).'%(filename_ncs))

        if not cascade:
            return



        #read data
        header_time_data = self.__mmap_ncs_packet_timestamps(filename_ncs)

        data = self.__mmap_ncs_data(filename_ncs)

        # ensure meaningful values for requested start and stop times
        # in case time is provided in samples: transform to absolute time units
        if isinstance(t_start,int):
            t_start = t_start / self.parameters_ncs[chid]['sampling_rate']
        if isinstance(t_stop,int):
            t_stop = t_stop / self.parameters_ncs[chid]['sampling_rate']

        # rescaling to global start time of recording (time of first sample in any file type)
        if t_start==None or t_start < (self.parameters_ncs[chid]['t_start'] - self.parameters_global['t_start']):
            t_start = (self.parameters_ncs[chid]['t_start'] - self.parameters_global['t_start'])

        if t_start > (self.parameters_ncs[chid]['t_stop'] - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is later than data are recorded (t_stop = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_ncs[chid]['t_stop'] - self.parameters_global['t_start']),
                                             filename_ncs))

        if t_stop==None or t_stop > (self.parameters_ncs[chid]['t_stop'] - self.parameters_global['t_start']):
            t_stop= (self.parameters_ncs[chid]['t_stop']  - self.parameters_global['t_start'])

        if t_stop < (self.parameters_ncs[chid]['t_start'] - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is earlier than data are recorded (t_start = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_ncs[chid]['t_start'] - self.parameters_global['t_start']),
                                             filename_ncs))
        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s) '
                             'for file %s.'%(t_start,t_stop,filename_ncs))

        # Extracting data signal in requested time window
        unit = pq.dimensionless # default value
        if lazy:
            sig = []
            p_id_start = 0
        else:

            tstamps = header_time_data * self.ncs_time_unit - self.parameters_global['t_start']

            #find data packet to start with signal construction
            starts = np.where(tstamps<=t_start)[0]
            if len(starts) == 0:
                self._diagnostic_print('Requested AnalogSignalArray not present in this time interval.')
                return
            else:
                #first packet to be included into signal
                p_id_start = starts[-1]
            #find data packet where signal ends (due to gap or t_stop)
            stops = np.where(tstamps>=t_stop)[0]
            if len(stops) !=0:
                first_stop = [stops[0]]
            else: first_stop = []

            # last packet to be included in signal
            p_id_stop = min(first_stop +  [len(data)])

            # search gaps in recording in time range to load
            gap_packets = [gap_id[0] for gap_id in self.parameters_ncs[chid]['gaps']  if gap_id[0]>p_id_start]
            if len(gap_packets)>0 and min(gap_packets) < p_id_stop:
                p_id_stop = min(gap_packets)
                warnings.warn('Analogsignalarray was shortened due to gap in recorded data '
                              ' of file %s at packet id %i'%(filename_ncs,min(gap_packets)))

            # search broken packets in time range to load
            broken_packets = []
            if 'broken_packet' in self.parameters_ncs[chid]:
                broken_packets = [packet[0] for packet in self.parameters_ncs[chid]['broken_packet'] \
                                  if packet[0]>p_id_start]
            if  len(broken_packets)>0 and  min(broken_packets) < p_id_stop:
                p_id_stop = min(broken_packets)
                warnings.warn('Analogsignalarray was shortened due to broken data packet in recorded data '
                              ' of file %s at packet id %i'%(filename_ncs,min(broken_packets)))

            # construct signal in valid packet range
            sig = np.array(data[p_id_start:p_id_stop+1],dtype=float)
            sig = sig.reshape(len(sig)*len(sig[0]))

            # ADBitVolts is not guaranteed to be present in the header!
            if 'ADBitVolts' in self.parameters_ncs[chid]:
            #TODO: Check transformation of recording signal into physical signal!
                sig *= self.parameters_ncs[chid]['ADBitVolts']
                unit = pq.V
            else:
                warnings.warn('Could not transform data from file %s into physical signal. '
                              'Missing "ADBitVolts" value in text header.')


        #defining sampling rate for rescaling purposes
        sampling_rate = self.parameters_ncs[chid]['sampling_unit']
        #creating neo AnalogSignalArray containing data
        anasig = AnalogSignalArray(signal = pq.Quantity(sig,unit, copy = False),
                                                    sampling_rate = 1*sampling_rate,
                                                    # rescaling t_start to sampling time units
                                                    t_start = (header_time_data[p_id_start] * self.ncs_time_unit - self.parameters_global['t_start']).rescale(1/sampling_rate),
                                                    name = 'channel_%i'%(chid),
                                                    channel_index = chid)

        # removing protruding parts of first and last data packet
        if anasig.t_start < t_start.rescale(anasig.t_start.units):
            anasig = anasig.time_slice(t_start.rescale(anasig.t_start.units),None)
        if anasig.t_stop > t_stop.rescale(anasig.t_start.units):
            anasig = anasig.time_slice(None,t_stop.rescale(anasig.t_start.units))

        anasig.annotations = self.parameters_ncs[chid]
        anasig.annotations['electrode_id'] = chid
        # this annotation is necesary for automatic genereation of recordingchannels
        anasig.annotations['channel_index'] = chid

        seg.analogsignalarrays.append(anasig)



    def read_nev(self, filename_nev, seg, lazy=False, cascade=True, t_start=None, t_stop=None):
        '''
        Reads associated nev file and attaches its content as eventarray to
        provided neo segment. In constrast to read_ncs times can not be provided
        in number of samples as a nev file has no inherent sampling rate.

        Arguments:
            filename_nev : Name of the .nev file to be loaded.
            seg : Neo Segment, to which the EventArray containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            EventArray. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time (quantity) that the EventArrays begin.
                            Default None.
            t_stop : time (quantity) that the EventArray end.
                            Default None.

        Returns:
            None

        Usage:
            TODO
        '''

        if filename_nev[-4:]!='.nev':
            filename_nev += '.nev'
        if sep in filename_nev:
            filename_nev = filename_nev.split(sep)[-1]

        if filename_nev not in self.nev_asso:
            raise ValueError('NeuralynxIO is attempting to read a file '
                            'not associated to this session (%s).'%(filename_nev))

        # # ensure meaningful values for requested start and stop times
        # # providing time is samples for nev file does not make sense as we don't know the underlying sampling rate
        if isinstance(t_start,int):
            raise ValueError('Requesting event information from nev file in samples does not make sense. '
                             'Requested t_start %s'%t_start)
        if isinstance(t_stop,int):
            raise ValueError('Requesting event information from nev file in samples does not make sense. '
                             'Requested t_stop %s'%t_stop)

        # ensure meaningful values for requested start and stop times
        if t_start==None or t_start < (self.parameters_nev[filename_nev]['t_start'] - self.parameters_global['t_start']):
            t_start = (self.parameters_nev[filename_nev]['t_start'] - self.parameters_global['t_start'])

        if t_start > (self.parameters_nev[filename_nev]['t_stop'] - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is later than data are recorded (t_stop = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_nev[filename_nev]['t_stop']  - self.parameters_global['t_start']),
                                             filename_nev))

        if t_stop==None or t_stop > (self.parameters_nev[filename_nev]['t_stop'] - self.parameters_global['t_start']):
            t_stop= (self.parameters_nev[filename_nev]['t_stop']  - self.parameters_global['t_start'])

        if t_stop < (self.parameters_nev[filename_nev]['t_start'] - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is earlier than data are recorded (t_start = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_nev[filename_nev]['t_start'] - self.parameters_global['t_start']),
                                             filename_nev))


        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s) '
                             'for file %s.'%(t_start,t_stop,filename_nev))


        data = self.__mmap_nev_file(filename_nev)
        # Extracting all events for one event type and put it into an event array
        # TODO: Check if this is the correct way of event creation.
        # TODO: (or should there be only a single event array for all event types?)
        for event_type in self.parameters_nev[filename_nev]['event_types']:
            # Extract all time stamps of digital markers and rescaling time
            type_mask = [i for i in range(len(data)) if (data[i][4]==event_type['event_id']
                                                         and data[i][5]==event_type['nttl']
                                                         and data[i][10]==event_type['name'])]
            marker_times = [t[3] for t in data[type_mask]] * self.nev_time_unit - self.parameters_global['t_start']



            #only consider Events in the requested time window (t_start, t_stop)
            time_mask = [i for i in range(len(marker_times)) if (marker_times[i]>=t_start and marker_times[i]<t_stop)]
            marker_times = marker_times[time_mask]

            # Do not create an eventarray if there are no events of this type in the requested time range
            if len(marker_times) == 0:
                continue

            ev = EventArray(times=pq.Quantity(marker_times, units=self.nev_time_unit, dtype="int"),
                                labels= event_type['name'],
                                name="Digital Marker " + str(event_type),
                                file_origin=filename_nev,
                                marker_id=event_type['event_id'],
                                digital_marker=True,
                                analog_marker=False,
                                nttl = event_type['nttl'])

            seg.eventarrays.append(ev)


    def read_nse(self, filename_nse, seg, lazy=False, cascade=True, t_start=None, t_stop=None, units=[],
                     waveforms = False):
        '''
        Reads nse file and attaches content as spike train to provided neo segment.
        Times can be provided in samples (integer values). If the nse file does not
        contain a sampling rate value, the ncs sampling rate on the same electrode
        is used.

        Arguments:
            filename_nse : Name of the .nse file to be loaded.
            seg : Neo Segment, to which the Spiketrain containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            SpikeTrain. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time or sample (quantity or integer) that the SpikeTrain begins.
                            Default None.
            t_stop : time or sample (quantity or integer) that the SpikeTrain ends.
                            Default None.
            units : unit ids to be loaded. If [], all units are loaded. Default [].
            waveforms : Load the waveform (up to 32 data points) for each
                            spike time. Default: False

        Returns:
            None

        Usage:
            TODO
        '''

        if filename_nse[-4:]!='.nse':
            filename_nse += '.nse'
        if sep in filename_nse:
            filename_nse = filename_nse.split(sep)[-1]

        # extracting channel id of requested file
        channel_id = self.get_channel_id_by_file_name(filename_nse)
        if channel_id != None:
            chid = channel_id
        else:
            #if nse file is empty it is not listed in self.parameters_nse, but
            # in self.nse_avail
            if filename_nse  in self.nse_avail:
                warnings.warn('NeuralynxIO is attempting to read an empty '
                            '(not associated) nse file (%s). '
                            'Not loading nse file.'%(filename_nse))
                return
            else:
                raise ValueError('NeuralynxIO is attempting to read a file '
                          'not associated to this session (%s).'%(filename_nse))


        # ensure meaningful values for requested start and stop times
        # in case time is provided in samples: transform to absolute time units
        # ncs sampling rate is best guess if there is no explicit sampling rate given for nse values.
        if 'sampling_rate' in self.parameters_nse[chid]:
            sr = self.parameters_nse[chid]['sampling_rate']
        elif chid in self.parameters_ncs and 'sampling_rate' in self.parameters_ncs[chid]:
            sr = self.parameters_ncs[chid]['sampling_rate']
        else:
            raise ValueError('No sampling rate present for channel id %i in nse file %s. '
                             'Could also not find the sampling rate of the respective ncs file.'%(chid,filename_nse))

        if isinstance(t_start,int):
            t_start = t_start / sr
        if isinstance(t_stop,int):
            t_stop = t_stop / sr

        # + rescaling global recording start (first sample in any file type)


        # This is not optimal, as there is no way to know how long the recording lasted after last spike
        if t_start==None or t_start < (self.parameters_nse[chid]['t_first'] - self.parameters_global['t_start']):
            t_start = (self.parameters_nse[chid]['t_first'] - self.parameters_global['t_start'])

        if t_start > (self.parameters_nse[chid]['t_last']  - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is later than data are recorded (t_stop = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_nse[chid]['t_last']  - self.parameters_global['t_start']),
                                             filename_nse))

        if t_stop==None:
            t_stop= (sys.maxsize) *self.nse_time_unit
        if t_stop==None or t_stop > (self.parameters_nse[chid]['t_last'] - self.parameters_global['t_start']):
            t_stop= (self.parameters_nse[chid]['t_last']  - self.parameters_global['t_start'])

        if t_stop < (self.parameters_nse[chid]['t_first'] - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is earlier than data are recorded (t_start = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_nse[chid]['t_first'] - self.parameters_global['t_start']),
                                             filename_nse))

        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s) '
                             'for file %s.'%(t_start,t_stop,filename_nse))


        # reading data
        [timestamps, channel_ids, cell_numbers, features, data_points] = self.__mmap_nse_packets(filename_nse)

        # load all units available if units==[]
        if units == []:
            units = np.unique(cell_numbers)
        elif not any([u in cell_numbers for u in units]):
            self._diagnostic_print('None of the requested unit ids (%s) present '
                                   'in nse file %s (contains units %s)'%(units,filename_nse,np.unique(cell_numbers)))

        # extracting spikes unit-wise and generate spiketrains
        for unit_i in units:
            if not lazy:
                # Extract all time stamps of that neuron on that electrode
                unit_mask = np.where(cell_numbers==unit_i)[0]
                spike_times = timestamps[unit_mask] * self.nse_time_unit
                spike_times= spike_times - self.parameters_global['t_start']
                time_mask = np.where(np.logical_and(spike_times>=t_start, spike_times < t_stop))
                spike_times = spike_times[time_mask]
            else:
                spike_times= pq.Quantity([], units=self.nse_time_unit)

            # Create SpikeTrain object
            st = SpikeTrain(times=spike_times,
                                t_start=t_start,
                                t_stop=t_stop,
                                sampling_rate=self.parameters_ncs[chid]['sampling_rate'],
                                name= "Channel %i, Unit %i"%(chid, unit_i),
                                file_origin=filename_nse,
                                unit_id=unit_i,
                                channel_id=chid)

            if waveforms and not lazy:
                # Collect all waveforms of the specific unit
                # For computational reasons: no units, no time axis
                st.waveforms = data_points[unit_mask][time_mask]
                # TODO: Add units to waveforms (pq.uV?) and add annotation left_sweep = x * pq.ms indicating when threshold crossing occurred in waveform

            st.annotations = self.parameters_nse[chid]
            st.annotations['electrode_id'] = chid
            # This annotations is necessary for automatic generation of recordingchannels
            st.annotations['channel_index'] = chid

            seg.spiketrains.append(st)



    def read_ntt(self, filename_ntt, seg, lazy=False, cascade=True, t_start=None, t_stop=None, units=[],
                     waveforms = False):
        '''
        Reads ntt file and attaches content as spike train to provided neo segment.

        Arguments:
            filename_ntt : Name of the .ntt file to be loaded.
            seg : Neo Segment, to which the Spiketrain containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            SpikeTrain. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time (quantity) that the SpikeTrain begins. Default None.
            t_stop : time (quantity) that the SpikeTrain ends. Default None.
            units : unit ids to be loaded. If [], all units are loaded. Default [].
            waveforms : Load the waveform (up to 32 data points) for each
                            spike time. Default: False

        Returns:
            None

        Usage:
            TODO
        '''

        if filename_ntt[-4:]!='.ntt':
            filename_ntt += '.ntt'
        if sep in filename_ntt:
            filename_ntt = filename_ntt.split(sep)[-1]

        # extracting channel id of requested file
        channel_id = self.get_channel_id_by_file_name(filename_ntt)
        if channel_id != None:
            chid = channel_id
        else:
            #if ntt file is empty it is not listed in self.parameters_ntt, but
            # in self.ntt_avail
            if filename_ntt  in self.ntt_avail:
                warnings.warn('NeuralynxIO is attempting to read an empty '
                            '(not associated) ntt file (%s). '
                            'Not loading ntt file.'%(filename_ntt))
                return
            else:
                raise ValueError('NeuralynxIO is attempting to read a file '
                          'not associated to this session (%s).'%(filename_ntt))


        # ensure meaningful values for requested start and stop times
        # in case time is provided in samples: transform to absolute time units
        # ncs sampling rate is best guess if there is no explicit sampling rate given for ntt values.
        if 'sampling_rate' in self.parameters_ntt[chid]:
            sr = self.parameters_ntt[chid]['sampling_rate']
        elif chid in self.parameters_ncs and 'sampling_rate' in self.parameters_ncs[chid]:
            sr = self.parameters_ncs[chid]['sampling_rate']
        else:
            raise ValueError('No sampling rate present for channel id %i in ntt file %s. '
                             'Could also not find the sampling rate of the respective ncs file.'%(chid,filename_ntt))

        if isinstance(t_start,int):
            t_start = t_start / sr
        if isinstance(t_stop,int):
            t_stop = t_stop / sr

        # + rescaling to global recording start (first sample in any recording file)
        if t_start==None or t_start < (self.parameters_ntt[chid]['t_first'] - self.parameters_global['t_start']):
            t_start = (self.parameters_ntt[chid]['t_first'] - self.parameters_global['t_start'])

        if t_start > (self.parameters_ntt[chid]['t_last']  - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is later than data are recorded (t_stop = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_ntt[chid]['t_last']  - self.parameters_global['t_start']),
                                             filename_ntt))

        if t_stop==None:
            t_stop= (sys.maxsize) *self.ntt_time_unit
        if t_stop==None or t_stop > (self.parameters_ntt[chid]['t_last'] - self.parameters_global['t_start']):
            t_stop= (self.parameters_ntt[chid]['t_last']  - self.parameters_global['t_start'])

        if t_stop < (self.parameters_ntt[chid]['t_first'] - self.parameters_global['t_start']):
            raise ValueError('Requested times window (%s to %s) is earlier than data are recorded (t_start = %s) '
                             'for file %s.'%(t_start,t_stop,
                                             (self.parameters_ntt[chid]['t_first'] - self.parameters_global['t_start']),
                                             filename_ntt))

        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s) '
                             'for file %s.'%(t_start,t_stop,filename_ntt))


        # reading data
        [timestamps, channel_ids, cell_numbers, features, data_points] = self.__mmap_ntt_packets(filename_ntt)

        #TODO: When ntt available: Implement 1 RecordingChannelGroup per Tetrode, such that each electrode gets its own recording channel

        # load all units available if units==[]
        if units == []:
            units = np.unique(cell_numbers)
        elif not any([u in cell_numbers for u in units]):
            self._diagnostic_print('None of the requested unit ids (%s) present '
                                   'in ntt file %s (contains units %s)'%(units,filename_ntt,np.unique(cell_numbers)))

        # loading data for each unit and generating spiketrain
        for unit_i in units:
            if not lazy:
                # Extract all time stamps of that neuron on that electrode
                mask = np.where(cell_numbers==unit_i)[0]
                spike_times = timestamps[mask] * self.ntt_time_unit
                spike_times= spike_times - self.parameters_global['t_start']
                spike_times = spike_times[np.where(np.logical_and(spike_times>=t_start, spike_times < t_stop))]
            else:
                spike_times= pq.Quantity([], units=self.ntt_time_unit)

            # Create SpikeTrain object
            st = SpikeTrain(times=spike_times,
                                t_start=t_start,
                                t_stop=t_stop,
                                sampling_rate=self.parameters_ncs[chid]['sampling_rate'],
                                name= "Channel %i, Unit %i"%(chid, unit_i),
                                file_origin=filename_ntt,
                                unit_id=unit_i,
                                channel_id=chid)

            # Collect all waveforms of the specific unit
            if waveforms and not lazy:
                # For computational reasons: no units, no time axis
                # transposing to adhere to neo guidline, which states that time should be in the first axis.
                # This is stupid and not intuitive.
                st.waveforms = np.array([data_points[t,:,:] for t in range(len(timestamps))
                                                if cell_numbers[t]==unit_i]).transpose()
                # TODO: Add units to waveforms (pq.uV?) and add annotation left_sweep = x * pq.ms indicating when threshold crossing occurred in waveform

            st.annotations = self.parameters_ntt[chid]
            st.annotations['electrode_id'] = chid
            # This annotations is necessary for automatic generation of recordingchannels
            st.annotations['channel_index'] = chid

            seg.spiketrains.append(st)


############# private routines #################################################



    def _associate(self, cachedir = None, usecache = 'hash'):
        """
        Associates the object with a specified Neuralynx session, i.e., a
        combination of a .nse, .nev and .ncs files. The meta data is read into the
        object for future reference.

        Arguments:
            cachedir : Directory for loading and saving hashes of recording sessions
                             and pickled meta information about files extracted during
                             association process
            use_cache: method used for cache identification. Possible values: 'hash'/
                            'always'/'datesize'/'never'. Default 'hash'
        Returns:
            -
        """

        # If already associated, disassociate first
        if self.associated:
            raise IOError("Trying to associate an already associated NeuralynxIO object.")

        # Create parameter containers
        # Dictionary that holds different parameters read from the .nev file
        self.parameters_nse = {}
        # List of parameter dictionaries for all potential file types
        self.parameters_ncs = {}
        self.parameters_nev = {}
        self.parameters_ntt = {}

        # combined global parameters
        self.parameters_global = {}

        # Scanning session directory for recorded files
        self.sessionfiles = [ f for f in listdir(self.sessiondir) if isfile(os.path.join(self.sessiondir,f)) ]

        # Listing available files
        self.ncs_avail = []
        self.nse_avail = []
        self.nev_avail = []
        self.ntt_avail = []

        # Listing associated (=non corrupted, non empty files)
        self.ncs_asso = []
        self.nse_asso = []
        self.nev_asso = []
        self.ntt_asso = []

        if usecache not in ['hash','always','datesize','never']:
            raise ValueError("Argument value of usecache '%s' is not valid. Accepted values are 'hash','always','datesize','never'"%usecache)

        if cachedir == None and usecache != 'never':
            raise ValueError('No cache directory provided.')

        # check if there are any changes of the data files -> new data check run
        check_files = True if usecache != 'always' else False # never checking files if usecache=='always'
        if cachedir != None and usecache != 'never':

            self._diagnostic_print('Calculating %s of session files to check for cached parameter files.'%usecache)
            cachefile = cachedir + sep + self.sessiondir.split(sep)[-1] + '/hashkeys'
            if not os.path.exists(cachedir + sep + self.sessiondir.split(sep)[-1]):
                os.makedirs(cachedir + sep + self.sessiondir.split(sep)[-1])

            if usecache=='hash':
                # calculates hash of all available files
                hashes_calc = {f:self.hashfile(open(self.sessiondir + sep + f, 'rb'), hashlib.sha256()) for f in self.sessionfiles}
            elif usecache=='datesize':
                hashes_calc = {f:self.datesizefile(self.sessiondir + sep + f) for f in self.sessionfiles}

            # load hashes saved for this session in an earlier loading run
            if os.path.exists(cachefile):
                hashes_read = pickle.load(open(cachefile, 'rb') )
            else: hashes_read = {}

            # compare hashes to previously saved meta data und load meta data if no changes occured
            if usecache == 'always' or all([f in hashes_calc and f in hashes_read and hashes_calc[f] == hashes_read[f] for f in self.sessionfiles]):
                check_files = False
                self._diagnostic_print('Using cached metadata from earlier analysis run in file %s. Skipping file checks.'%cachefile)

                # loading saved parameters
                parameterfile = cachedir + sep + self.sessiondir.split(sep)[-1] + '/parameters.cache'
                if os.path.exists(parameterfile):
                    parameters_read = pickle.load(open(parameterfile, 'rb') )
                else:
                    raise IOError('Inconsistent cache files.')

                for IOdict, dictname in [(self.parameters_global,'global'),
                                         (self.parameters_ncs,'ncs'),
                                         (self.parameters_nse,'nse'),
                                         (self.parameters_nev,'nev'),
                                         (self.parameters_ntt,'ntt')]:
                    IOdict.update(parameters_read[dictname])


        for filename in self.sessionfiles:
            # Extracting only continuous signal files (.ncs)
            if filename[-4:] == '.ncs':
                self.ncs_avail.append(filename)
            elif filename[-4:] == '.nse':
                self.nse_avail.append(filename)
            elif filename[-4:] == '.nev':
                self.nev_avail.append(filename)
            elif filename[-4:] == '.ntt':
                self.ntt_avail.append(filename)
            else:
                self._diagnostic_print('Ignoring file of unknown data type %s'%filename)

        if check_files:
            self._diagnostic_print('Starting individual file checks.')
            #=======================================================================
            # # Scan NCS files
            #=======================================================================

            self._diagnostic_print('\nDetected %i .ncs file(s).'%(len(self.ncs_avail)))

            for ncs_file in self.ncs_avail:
                # Loading individual NCS file and extracting parameters
                self._diagnostic_print("Scanning " + ncs_file + ".")

                # Reading file packet headers
                filehandle = self.__mmap_ncs_packet_headers(ncs_file)
                if filehandle == None:
                    continue

                try:
                    # Checking consistency of ncs file
                    self.__ncs_packet_check(filehandle)
                except AssertionError:
                    warnings.warn('Session file %s did not pass data packet check. '
                                  'This file can not be loaded.'%ncs_file)
                    continue

                # Reading data packet header information and store them in parameters_ncs
                self.__read_ncs_data_headers(filehandle, ncs_file)

                # Reading txt file header
                channel_id = self.get_channel_id_by_file_name(ncs_file)
                self.__read_text_header(ncs_file,self.parameters_ncs[channel_id])

                # Check for invalid starting times of data packets in ncs file
                self.__ncs_invalid_first_sample_check(filehandle)

                # Check ncs file for gaps
                self.__ncs_gap_check(filehandle)

                self.ncs_asso.append(ncs_file)

            #=======================================================================
            # # Scan NSE files
            #=======================================================================

            # Loading individual NSE file and extracting parameters
            self._diagnostic_print('\nDetected %i .nse file(s).'%(len(self.nse_avail)))

            for nse_file in self.nse_avail:
                # Loading individual NSE file and extracting parameters
                self._diagnostic_print('Scanning ' + nse_file + '.')

                # Reading file
                filehandle = self.__mmap_nse_packets(nse_file)
                if filehandle == None:
                    continue

                try:
                    # Checking consistency of nse file
                    self.__nse_check(filehandle)
                except AssertionError:
                    warnings.warn('Session file %s did not pass data packet check. '
                                  'This file can not be loaded.'%nse_file)
                    continue

                # Reading header information and store them in parameters_nse
                self.__read_nse_data_header(filehandle, nse_file)

                # Reading txt file header
                channel_id = self.get_channel_id_by_file_name(nse_file)
                self.__read_text_header(nse_file,self.parameters_nse[channel_id])

                # using sampling rate from txt header, as this is not saved in data packets
                if 'SamplingFrequency' in self.parameters_nse[channel_id]:
                    self.parameters_nse[channel_id]['sampling_rate'] = \
                        (self.parameters_nse[channel_id]['SamplingFrequency'] * self.nse_sr_unit)

                self.nse_asso.append(nse_file)

            #=======================================================================
            # # Scan NEV files
            #=======================================================================

            self._diagnostic_print('\nDetected %i .nev file(s).'%(len(self.nev_avail)))

            for nev_file in self.nev_avail:
                # Loading individual NEV file and extracting parameters
                self._diagnostic_print('Scanning ' + nev_file + '.')

                # Reading file
                filehandle = self.__mmap_nev_file(nev_file)
                if filehandle == None:
                    continue

                try:
                    # Checking consistency of nev file
                    self.__nev_check(filehandle)
                except AssertionError:
                    warnings.warn('Session file %s did not pass data packet check. '
                                  'This file can not be loaded.'%nev_file)
                    continue

                # Reading header information and store them in parameters_nev
                self.__read_nev_data_header(filehandle, nev_file)

                # Reading txt file header
                self.__read_text_header(nev_file,self.parameters_nev[nev_file])

                self.nev_asso.append(nev_file)

            #=======================================================================
            # # Scan NTT files
            #=======================================================================

            self._diagnostic_print('\nDetected %i .ntt file(s).'%(len(self.ntt_avail)))

            for ntt_file in self.ntt_avail:
                # Loading individual NTT file and extracting parameters
                self._diagnostic_print('Scanning ' + ntt_file + '.')

                # Reading file
                filehandle = self.__mmap_ntt_file(ntt_file)
                if filehandle == None:
                    continue

                try:
                    # Checking consistency of nev file
                    self.__ntt_check(filehandle)
                except AssertionError:
                    warnings.warn('Session file %s did not pass data packet check. '
                                  'This file can not be loaded.'%ntt_file)
                    continue

                # Reading header information and store them in parameters_nev
                self.__read_ntt_data_header(filehandle, ntt_file)

                # Reading txt file header
                self.__read_ntt_text_header(ntt_file)

                # using sampling rate from txt header, as this is not saved in data packets
                if 'SamplingFrequency' in self.parameters_ntt[channel_id]:
                    self.parameters_ntt[channel_id]['sampling_rate'] = \
                        (self.parameters_ntt[channel_id]['SamplingFrequency'] * self.ntt_sr_unit)

                self.ntt_asso.append(ntt_file)

            #=======================================================================
            # # Check consistency across files
            #=======================================================================

            # check RECORDING_OPENED / CLOSED times (from txt header) for different files
            for parameter_collection in [self.parameters_ncs, self.parameters_nse,
                                         self.parameters_nev, self.parameters_ntt]:
                # check recoding_closed times for specific file types
                if any(np.abs(np.diff([i['recording_opened'] for i in parameter_collection.values()]))
                                                                            >datetime.timedelta(seconds=0.1)):
                    raise ValueError('NCS files were opened for recording with a delay greater than 0.1 second.')

                # check recoding_closed times for specific file types
                if any(np.diff([i['recording_closed'] for i in parameter_collection.values()
                                        if i['recording_closed'] != None])>datetime.timedelta(seconds=0.1)):
                    raise ValueError('NCS files were closed after recording with a delay greater than 0.1 second.')

            # get maximal duration of any file in the recording
            parameter_collection = self.parameters_ncs.values() + self.parameters_nse.values() \
                                   + self.parameters_ntt.values() + self.parameters_nev.values()
            self.parameters_global['recording_opened'] = min([i['recording_opened']for i in parameter_collection])
            self.parameters_global['recording_closed'] = max([i['recording_closed']for i in parameter_collection])

            ############ Set up GLOBAL TIMING SCHEME #############################
            for file_type, parameter_collection in [('ncs',self.parameters_ncs), ('nse',self.parameters_nse),
                                                    ('nev',self.parameters_nev), ('ntt',self.parameters_ntt)]:
                # check starting times
                name_t1, name_t2 = ['t_start','t_stop'] if (file_type != 'nse' and file_type != 'ntt') \
                                                        else ['t_first','t_last']

                # checking if files of same type start at same time point
                if file_type != 'nse' and file_type != 'ntt' \
                        and len(np.unique(np.array([i[name_t1].magnitude for i in parameter_collection.values()]))) > 1:
                    raise ValueError('%s files do not start at same time point.'%file_type)

                # saving t_start and t_stop for each file type available
                if len([i[name_t1] for i in parameter_collection.values()]):
                    self.parameters_global['%s_t_start'%file_type] = min([i[name_t1]
                                                                          for i in parameter_collection.values()])
                    self.parameters_global['%s_t_stop'%file_type] = min([i[name_t2]
                                                                         for i in parameter_collection.values()])

            # extracting minimial t_start and maximal t_stop value for this recording session
            self.parameters_global['t_start'] = min([self.parameters_global['%s_t_start'%t]
                                                     for t in ['ncs','nev','nse','ntt']
                                                     if '%s_t_start'%t in self.parameters_global])
            self.parameters_global['t_stop'] = max([self.parameters_global['%s_t_stop'%t]
                                                    for t in ['ncs','nev','nse','ntt']
                                                    if '%s_t_start'%t in self.parameters_global])


            # checking gap consistency across ncs files
            #check number of gaps detected
            if len(np.unique([len(i['gaps']) for i in self.parameters_ncs.values()])) != 1:
                raise ValueError('NCS files contain different numbers of gaps!')
            # check consistency of gaps across files and create global gap collection
            self.parameters_global['gaps'] = []
            for g in range(len(self.parameters_ncs.values()[0]['gaps'])):
                integrated = False
                gap_stats = np.unique([i['gaps'][g] for i in self.parameters_ncs.values()],return_counts=True)
                if len(gap_stats[0]) != 3 or len(np.unique(gap_stats[1])) != 1:
                    raise ValueError('Gap number %i is not consistent across NCS files.'%(g))
                else:
                    # check if this is second part of already existing gap
                    for gg in range(len(self.parameters_global['gaps'])):
                        globalgap = self.parameters_global['gaps'][gg]
                        # check if stop time of first is start time of second -> continuous gap
                        if globalgap[2] == self.parameters_ncs.values()[0]['gaps'][g][1]:
                            self.parameters_global['gaps'][gg] = self.parameters_global['gaps'][gg][:2] + (self.parameters_ncs.values()[0]['gaps'][g][2],)
                            integrated = True
                            break

                    if not integrated:
                        # add as new gap if this is not a continuation of existing global gap
                        self.parameters_global['gaps'].append(self.parameters_ncs.values()[0]['gaps'][g])


        # save results of association for future analysis together with hash values for change tracking
        if cachedir != None and usecache!='never':
            pickle.dump( {'global': self.parameters_global,
                          'ncs': self.parameters_ncs,
                          'nev': self.parameters_nev,
                          'nse': self.parameters_nse,
                          'ntt': self.parameters_ntt},
                         open( cachedir + sep + self.sessiondir.split(sep)[-1] + '/parameters.cache', 'wb' ))
            if usecache != 'always':
                pickle.dump( hashes_calc, open(cachedir + sep + self.sessiondir.split(sep)[-1] + '/hashkeys', 'wb' ))

        self.associated = True




    #################### private routines #########################################################ü


    ################# Memory Mapping Methods

    def __mmap_nse_packets(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        filesize = getsize(self.sessiondir + sep + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + sep + filename,
                            dtype='<u2', shape = ((filesize-16384)/2/56,56),
                            mode='r', offset=16384)

            # reconstructing original data
            # first 4 ints -> timestamp in microsec
            timestamps = data[:,0] + data[:,1]*2**16 + data[:,2]*2**32 + data[:,3]*2**48
            channel_id = data[:,4] + data[:,5]*2**16
            cell_number = data[:,6] + data[:,7]*2**16
            features = [data[:,p] + data[:,p+1]*2**16 for p in range(8,23,2)]
            features = np.array(features,dtype='i4')

            data_points = data[:,24:56].astype('i2')
            del data
            return timestamps, channel_id, cell_number, features, data_points
        else: return None


    def __mmap_ncs_data(self,filename):
        """ Memory map of the Neuralynx .ncs file optimized for data extraction"""
        if getsize(self.sessiondir + sep + filename) > 16384:
            data = np.memmap(self.sessiondir + sep + filename, dtype=np.dtype(('i2',(522))),mode='r', offset=16384)
            #removing data packet headers and flattening data
            return data[:,10:]
        else: return None

    def __mmap_ncs_packet_headers(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        filesize = getsize(self.sessiondir + sep + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + sep + filename,
                            dtype='<u4', shape = ((filesize-16384)/4/261,261),
                            mode='r', offset=16384)

            ts = data[:,0:2]
            multi=np.repeat(np.array([1,2**32],ndmin=2),len(data),axis=0)
            timestamps=np.sum(ts*multi,axis=1)
            #timestamps = data[:,0] + (data[:,1] *2**32)
            header_u4 = data[:,2:5]

            return timestamps, header_u4
        else: return None

    def __mmap_ncs_packet_timestamps(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        filesize = getsize(self.sessiondir + sep + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + sep + filename,
                            dtype='<u4', shape = ((filesize-16384)/4/261,261),
                            mode='r', offset=16384)

            ts = data[:,0:2]
            multi=np.repeat(np.array([1,2**32],ndmin=2),len(data),axis=0)
            timestamps=np.sum(ts*multi,axis=1)
            # timestamps = data[:,0] + data[:,1]*2**32

            return timestamps
        else: return None


    def __mmap_nev_file(self, filename):
        """ Memory map the Neuralynx .nev file """
        nev_dtype = np.dtype([
            ('reserved', '<i2'),
            ('system_id', '<i2'),
            ('data_size', '<i2'),
            ('timestamp', '<u8'),
            ('event_id', '<i2'),
            ('ttl_input', '<i2'),
            ('crc_check', '<i2'),
            ('dummy1', '<i2'),
            ('dummy2', '<i2'),
            ('extra', '<i4',   (8,)),
            ('event_string', 'a128'),
        ])

        if getsize(self.sessiondir + sep + filename) > 16384:
            return np.memmap(self.sessiondir + sep + filename,
                                         dtype=nev_dtype, mode='r', offset=16384)
        else: return None

    def __mmap_ntt_file(self, filename):
        """ Memory map the Neuralynx .nse file """
        nse_dtype = np.dtype([
            ('timestamp', '<u8'),
            ('sc_number', '<u4'),
            ('cell_number', '<u4'),
            ('params', '<u4',   (8,)),
            ('data', '<i2', (32, 4)),
        ])
        if getsize(self.sessiondir + sep + filename) > 16384:
            return np.memmap(self.sessiondir + sep + filename,
                                         dtype=nse_dtype, mode='r', offset=16384)
        else: return None


    def __mmap_ntt_packets(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        filesize = getsize(self.sessiondir + sep + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + sep + filename,
                            dtype='<u2', shape = ((filesize-16384)/2/152,152),
                            mode='r', offset=16384)

            # reconstructing original data
            # first 4 ints -> timestamp in microsec
            timestamps = data[:,0] + data[:,1]*2**16 + data[:,2]*2**32 + data[:,3]*2**48
            channel_id = data[:,4] + data[:,5]*2**16
            cell_number = data[:,6] + data[:,7]*2**16
            features = [data[:,p] + data[:,p+1]*2**16 for p in range(8,23,2)]
            features = np.array(features,dtype='i4')

            data_points = data[:,24:152].astype('i2').reshape((4,32))
            del data
            return timestamps, channel_id, cell_number, features, data_points
        else: return None


    #___________________________ header extraction __________________________

    def __read_text_header(self,filename,parameter_dict):
        # Reading main file header (plain text, 16kB)
        text_header = open(self.sessiondir + sep + filename,'r').read(16384)
        #separating lines of header and ignoring last line (fill), check if Linux or Windows OS
        if sep == '/':
            text_header = text_header.split('\r\n')[:-1]
        if sep == '\\':
            text_header = text_header.split('\n')[:-1]

        # extracting filename and recording opening/closing time
        header_dict = self.__read_intro_txt_header(text_header)
        parameter_dict.update(header_dict)



        # minor parameters possibly saved in header (for any file type)
        minor_keys =  ['CheetahRev','AcqEntName','FileType','FileVersion','RecordSize',
                          'HardwareSubSystemName','HardwareSubSystemType',
                          'SamplingFrequency','ADMaxValue','ADBitVolts','NumADChannels',
                          'ADChannel','InputRange','InputInverted','DSPLowCutFilterEnabled',
                          'DspLowCutFrequency','DspLowCutNumTaps','DspLowCutFilterType',
                          'DSPHighCutFilterEnabled','DspHighCutFrequency','DspHighCutNumTaps',
                          'DspHighCutFilterType','DspDelayCompensation','DspFilterDelay_\xb5s',
                           'DisabledSubChannels','WaveformLength','AlignmentPt','ThreshVal',
                           'MinRetriggerSamples','SpikeRetriggerTime','DualThresholding',
                           'Feature Peak 0','Feature Valley 1','Feature Energy 2','Feature Height 3',
                           'Feature NthSample 4','Feature NthSample 5','Feature NthSample 6',
                           'Feature NthSample 7']


        #extracting minor key values of header (only taking into account non-empty lines)
        for i, minor_entry in enumerate([text for text in text_header[4:] if text != '']):
            matching_key = [key for key in minor_keys if minor_entry.strip('-').startswith(key)]
            if len(matching_key)==1:
                matching_key=matching_key[0]
                minor_value = minor_entry.split(matching_key)[1].strip(' ').rstrip(' ')

                # determine data type of entry
                if minor_value.isdigit():
                    # converting to int if possible
                    minor_value = int(minor_value)
                else:
                    # converting to float if possible
                    try:
                        minor_value = float(minor_value)
                    except:
                        pass

                if matching_key in parameter_dict:
                    warnings.warn('Multiple entries for %s in text header of %s'%(matching_key,filename))
                else:
                    parameter_dict[matching_key] = minor_value
            elif len(matching_key)>1:
                raise ValueError('Inconsistent minor key list for text header interpretation.')
            else:
                warnings.warn('Skipping text header entry %s, because it is not in minor key list'%minor_entry)

        self._diagnostic_print('Successfully decoded text header of file (%s).'%(filename))



    def __read_intro_txt_header(self,txt_header):
        """
        Reading first 3 lines of text header and extracting filename and recording openind/closing
        :param txt_header:
        :return: dictionary containing the original filename and recording opening / closing
        """
        output = {}
        try:
            # checking title of header
            if txt_header[0] != '######## Neuralynx Data File Header':
                raise TypeError('NCS file has unkown Neuralynx header title')

            filename_struct = re.compile('## File Name (?P<filename>.{1,})')
            # extracting filename
            filename_match = filename_struct.match(txt_header[1])
            if filename_match:
                output['recording_file_name'] = filename_match.groupdict().values()[0]

            # extracting datetime entries in header
            datetime_struct = re.compile('## Time (?P<mode>\S{6}) \(m/d/y\): '
                                         '(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})  '
                                         '\(h:m:s\.ms\) (?P<hour>\d{1,2}):(?P<minute>\d{1,2}):'
                                         '(?P<second>\d{1,2})\.(?P<millisecond>\d{1,3})')

            for line in [2,3]:
                datetime_match = datetime_struct.match(txt_header[line])
                if datetime_match:
                    datetime_dict = datetime_match.groupdict()
                    mode = datetime_dict.pop('mode').lower()
                    output['recording_' + mode] = datetime.datetime(int(datetime_dict['year']),
                                                                    int(datetime_dict['month']),
                                                                    int(datetime_dict['day']),
                                                                    int(datetime_dict['hour']),
                                                                    int(datetime_dict['minute']),
                                                                    int(datetime_dict['second']),
                                                                    1000*int(datetime_dict['millisecond']))
                elif txt_header[line].startswith('## Time Closed File was not closed properly'):
                    output['recording_closed'] = None
                    warnings.warn('Text header of file %s does not contain recording closed time. '
                                  'File was not closed properly.'%output['recording_file_name'])
                else:
                    raise TypeError('NCS file has unknown major parameters in header')

        except TypeError:
            warnings.warn('WARNING: NeuralynxIO is unable to extract data from text header! '
                             'Continue with loading data.')

        return output


    def __read_ncs_data_headers(self, filehandle, filename):
        '''
        Reads the .ncs data block headers and stores the information in the
        object's parameters_ncs dictionary.

        Args:
            filehandle (file object):
                Handle to the already opened .ncs file.
            filename (string):
                Name of the ncs file.
        Returns:
            dict of extracted data
        '''
        timestamps = filehandle[0]
        header_u4 = filehandle[1]

        channel_id = header_u4[0][0]
        sr = header_u4[0][1] # in Hz

        t_start = timestamps[0] # in microseconds
        #calculating corresponding time stamp of first sample, that was not recorded any more
        #t_stop= time of first sample in last packet +(#samples per packet * conversion factor / sampling rate)
        # conversion factor is needed as times are recorded in ms
        t_stop = timestamps[-1] + ((header_u4[-1][2]) * (1/self.ncs_time_unit.rescale(pq.s)).magnitude / header_u4[-1][1])

        if channel_id in self.parameters_ncs:
            raise ValueError('Detected multiple ncs files for channel_id %i.'%(channel_id))
        else:
            self.parameters_ncs[channel_id] = { 'filename':filename,
                                                't_start': t_start * self.ncs_time_unit,
                                                't_stop': t_stop * self.ncs_time_unit,
                                                'sampling_rate': sr * self.ncs_sr_unit,
                                                'sampling_unit': pq.CompoundUnit( '%f*%s'%(sr,self.ncs_sr_unit.symbol)),
                                                'gaps': []}

            return {channel_id: self.parameters_ncs[channel_id]}


    def __read_nse_data_header(self, filehandle, filename):
        '''
        Reads the .nse data block headers and stores the information in the
        object's parameters_ncs dictionary.

        Args:
            filehandle (file object):
                Handle to the already opened .nse file.
            filename (string):
                Name of the nse file.
        Returns:
            -
        '''

        [timestamps, channel_ids, cell_numbers, features, data_points] = filehandle

        if filehandle != None:

            t_first = timestamps[0] # in microseconds
            t_last = timestamps[-1] # in microseconds
            channel_id = channel_ids[0]
            cell_count = cell_numbers[0] #number of cells identified
            # spike_parameters = filehandle[0][3]
        # else:
        #     t_first = None
        #     channel_id = None
        #     cell_count = 0
        #     # spike_parameters =  None
        #
        #     self._diagnostic_print('Empty file: No information contained in %s'%filename)

            self.parameters_nse[channel_id] = { 'filename':filename,
                                                't_first': t_first * self.nse_time_unit,
                                                't_last': t_last * self.nse_time_unit,
                                                'cell_count': cell_count}

    def __read_ntt_data_header(self, filehandle, filename):
        '''
        Reads the .nse data block headers and stores the information in the
        object's parameters_ncs dictionary.

        Args:
            filehandle (file object):
                Handle to the already opened .nse file.
            filename (string):
                Name of the nse file.
        Returns:
            -
        '''

        [timestamps, channel_ids, cell_numbers, features, data_points] = filehandle

        if filehandle != None:

            t_first = timestamps[0] # in microseconds
            t_last = timestamps[-1] # in microseconds
            channel_id = channel_ids[0]
            cell_count = cell_numbers[0] #number of cells identified
            # spike_parameters = filehandle[0][3]
        # else:
        #     t_first = None
        #     channel_id = None
        #     cell_count = 0
        #     # spike_parameters =  None
        #
        #     self._diagnostic_print('Empty file: No information contained in %s'%filename)

            self.parameters_ntt[channel_id] = { 'filename':filename,
                                                't_first': t_first * self.ntt_time_unit,
                                                't_last': t_last * self.nse_time_unit,
                                                'cell_count': cell_count}

    def __read_nev_data_header(self, filehandle, filename):
        '''
        Reads the .nev data block headers and stores the relevant information in the
        object's parameters_nev dictionary.

        Args:
            filehandle (file object):
                Handle to the already opened .nev file.
            filename (string):
                Name of the nev file.
        Returns:
            -
        '''

        # Extracting basic recording events to be able to check recording consistency
        if filename in self.parameters_nev:
            raise ValueError('Detected multiple nev files of name %s.'%(filename))
        else:
            self.parameters_nev[filename] = {}
            if 'Starting_Recording' in self.parameters_nev[filename]:
                raise ValueError('Trying to read second nev file of name %s. '
                                 ' Only one can be handled.'%filename)
            self.parameters_nev[filename]['Starting_Recording'] = []
            self.parameters_nev[filename]['events'] = []
            for event in filehandle:
                # separately extracting 'Starting Recording'
                if event[4] == 11 and event[10] == 'Starting Recording':
                    self.parameters_nev[filename]['Starting_Recording'].append(event[3]*self.nev_time_unit)

                # adding all events to parameter collection
                self.parameters_nev[filename]['events'].append({'timestamp':event[3]*self.nev_time_unit,
                                                                'event_id':event[4],
                                                                'nttl':event[5],
                                                                'name':event[10]})

            if len(self.parameters_nev[filename]['Starting_Recording']) < 1:
                raise ValueError('No Event "Starting_Recording" detected in %s'%(filename))

            self.parameters_nev[filename]['t_start'] = min(self.parameters_nev[filename]['Starting_Recording'])
            # t_stop = time stamp of last event in file
            self.parameters_nev[filename]['t_stop'] = max([e['timestamp'] for e in
                                                                    self.parameters_nev[filename]['events']])

            # extract all occurring event types (= combination of nttl,event_id and name/string)
            event_types = copy.deepcopy(self.parameters_nev[filename]['events'])
            for d in event_types:
                d.pop('timestamp')
            self.parameters_nev[filename]['event_types'] = [dict(y) for y in set(tuple(x.items()) for x in event_types)]




    #________________ File Checks __________________________________

    def __ncs_packet_check(self,filehandle):
        '''
        Checks consistency of data in ncs file and raises assertion error if a
        check fails. Detected recording gaps are added to parameter_ncs

        Args:
            filehandle (file object):
                Handle to the already opened .ncs file.
        '''


        timestamps = filehandle[0]
        header_u4 = filehandle[1]


        # checking sampling rate of data packets
        sr0 = header_u4[0,1]
        assert all(header_u4[:,1] == sr0)

        # checking channel id of data packets
        channel_id = header_u4[0,0]
        assert all(header_u4[:,0] == channel_id)

        #time offset of data packets
        # TODO: Check if there is a safer way to do the delta_t check for ncs data packets
        # this is a not safe assumption, that the first two data packets have correct time stamps
        delta_t = timestamps[1] - timestamps[0]

        # valid samples of first data packet
        temp_valid_samples = header_u4[0,2]

        # unit test
        # time difference between packets corresponds to number of recorded samples
        assert delta_t == (temp_valid_samples / (self.ncs_time_unit.rescale(pq.s).magnitude * sr0))

        self._diagnostic_print('NCS packet check successful.')



    def __nse_check(self,filehandle):
        '''
        Checks consistency of data in ncs file and raises assertion error if a
        check fails.

        Args:
            filehandle (file object):
                Handle to the already opened .nse file.
        '''

        [timestamps, channel_ids, cell_numbers, features, data_points] = filehandle

        assert all(channel_ids == channel_ids[0])

        assert all([len(dp)==len(data_points[0]) for dp in data_points])

        self._diagnostic_print('NSE file check successful.')


    def __nev_check(self,filehandle):
        '''
        Checks consistency of data in nev file and raises assertion error if a
        check fails.

        Args:
            filehandle (file object):
                Handle to the already opened .nev file.
        '''

        # this entry should always equal 2 (see Neuralynx File Description), but it is not. For me, this is 0.
        assert all([f[2]==2 or f[2]==0 for f in filehandle])

        # TODO: check with more nev files, if index 0,1,2,6,7,8 and 9 can be non-zero. Meaning? Include in event extraction.
        # only observed 0 for index 0,1,2,6,7,8,9 in nev files.
        # If they are non-zero, this needs to be included in event extraction
        assert all([f[0]==0 for f in filehandle])
        assert all([f[1]==0 for f in filehandle])
        assert all([f[2]==0 for f in filehandle])

        assert all([f[6]==0 for f in filehandle])
        assert all([f[7]==0 for f in filehandle])
        assert all([f[8]==0 for f in filehandle])
        assert all([all(f[9]==0) for f in filehandle])

        self._diagnostic_print('NEV file check successful.')


    def __ntt_check(self,filehandle):
        '''
        Checks consistency of data in ncs file and raises assertion error if a
        check fails.

        Args:
            filehandle (file object):
                Handle to the already opened .nse file.
        '''
        # TODO: check this when first .ntt files are available
        [timestamps, channel_ids, cell_numbers, features, data_points] = filehandle

        assert all(channel_ids == channel_ids[0])

        assert all([len(dp)==len(data_points[0]) for dp in data_points])

        self._diagnostic_print('NTT file check successful.')


    def __ncs_gap_check(self,filehandle):
        '''
        Checks individual data blocks of ncs files for consistent starting times with respect to sample count.
        This covers intended recording gaps as well as shortened data packet, which are incomplete
        '''

        timestamps = filehandle[0]
        header_u4 = filehandle[1]
        channel_id = header_u4[0,0]
        if channel_id not in self.parameters_ncs:
            self.parameters_ncs[channel_id] = {}

        #time stamps of data packets
        delta_t = timestamps[1] - timestamps[0] #in microsec
        data_packet_offsets = np.diff(timestamps) # in microsec

        # check if delta_t corresponds to number of valid samples present in data packets
        # NOTE: This also detects recording gaps!
        valid_samples = header_u4[:-1,2]
        sampling_rate = header_u4[0,1]
        packet_checks = (valid_samples/(self.ncs_time_unit.rescale(pq.s).magnitude*sampling_rate))==data_packet_offsets
        if not all(packet_checks):
            if 'broken_packets' not in self.parameters_ncs[channel_id]:
                self.parameters_ncs[channel_id]['broken_packets'] = []
            broken_packets = np.where(np.array(packet_checks)==False)[0]
            for broken_packet in broken_packets:
                self.parameters_ncs[channel_id]['broken_packets'].append((broken_packet,
                                                                         valid_samples[broken_packet],
                                                                         data_packet_offsets[broken_packet]))
                self._diagnostic_print('Detected broken packet in NCS file at '
                                        'packet id %i (sample number %i '
                                        'time offset id %i)'  %(broken_packet,
                                                         valid_samples[broken_packet],
                                                         data_packet_offsets[broken_packet])) # in microsec


        #checking for irregular data packet durations -> gaps / shortened data packets
        if not all(data_packet_offsets == delta_t):
            if 'gaps' not in self.parameters_ncs[channel_id]:
                self.parameters_ncs[channel_id]['gaps'] = []
            # gap identification by (sample of gap start, duration)
            # gap packets
            gap_packet_ids = np.where(data_packet_offsets != delta_t)[0]
            for gap_packet_id in gap_packet_ids:

                #skip if this packet starting time is known to be corrupted
                # hoping no corruption and gap occurrs simultaneously #TODO: Check this
                # corrupted time stamp affects two delta_t comparisons:
                if gap_packet_id in self.parameters_ncs[channel_id]['invalid_first_samples'] \
                    or gap_packet_id + 1 in self.parameters_ncs[channel_id]['invalid_first_samples']:
                    continue

                gap_start = timestamps[gap_packet_id] # t_start of last packet [microsec]
                gap_stop = timestamps[gap_packet_id+1] # t_stop of first packet [microsec]

                self.parameters_ncs[channel_id]['gaps'].append((gap_packet_id,gap_start,gap_stop)) #[,microsec,microsec]
                self._diagnostic_print('Detected gap in NCS file between'
                                        'sample time %i and %i  (last correct '
                                        'packet id %i)'  %(gap_start,gap_stop,
                                                            gap_packet_id))


    def __ncs_invalid_first_sample_check(self,filehandle):
        '''
        Checks data blocks of ncs files for corrupted starting times indicating
        a missing first sample in the data packet. These are then excluded from
        the gap check, but ignored for further analysis.
        '''
        timestamps = filehandle[0]
        header_u4 = filehandle[1]
        channel_id = header_u4[0,0]
        self.parameters_ncs[channel_id]['invalid_first_samples'] = []

        #checking if first bit of timestamp is 1, which indicates error
        invalid_packet_ids = np.where(timestamps >= 2**55)[0]
        if len(invalid_packet_ids)>0:
            warnings.warn('Invalid first sample(s) detected in ncs file'
                            '(packet id(s) %i)! This error is ignored in'
                            'subsequent routines.'%(invalid_packet_ids))
            self.parameters_ncs[channel_id]['invalid_first_samples'] = invalid_packet_ids

            #checking consistency of data around corrupted packet time
            for invalid_packet_id in invalid_packet_ids:
                if invalid_packet_id < 2 or invalid_packet_id > len(filehandle) -2:
                    raise ValueError('Corrupted ncs data packet at the beginning'
                                        'or end of file.')
                elif (timestamps[invalid_packet_id+1] - timestamps[invalid_packet_id-1]
                != 2* (timestamps[invalid_packet_id-1] - timestamps[invalid_packet_id-2])):
                    raise ValueError('Starting times of ncs data packets around'
                                     'corrupted data packet are not consistent!')


    ############ Supplementory Functions ###########################
    def get_channel_id_by_file_name(self,filename):
        """
        Checking parameters of NCS, NSE and NTT Files for given filename and return channel_id if result is consistent
        :param filename:
        :return:
        """
        channel_ids = []
        channel_ids += [k for k in self.parameters_ncs if self.parameters_ncs[k]['filename'] == filename]
        channel_ids += [k for k in self.parameters_nse if self.parameters_nse[k]['filename'] == filename]
        channel_ids += [k for k in self.parameters_ntt if self.parameters_ntt[k]['filename'] == filename]
        if len(np.unique(np.asarray(channel_ids))) == 1:
            return  channel_ids[0]
        elif len(channel_ids) > 1:
            raise ValueError('Ambiguous channel ids detected. Filename %s is associated to different channels of '
            'NCS and NSE and NTT %s'%(filename,channel_ids))
        else: # if filename was not detected
            return None


    def hashfile(self,afile, hasher, blocksize=65536):
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
        return hasher.digest()

    def datesizefile(self,filename):
        return str(os.path.getmtime(filename)) + '_' +  str(os.path.getsize(filename))


    def _diagnostic_print(self, text):
        '''
        Print a diagnostic message.

        Args:
            text (string):
                Diagnostic text to print.

        Returns:
            -
        '''

        if self._print_diagnostic:
            print('NeuralynxIO: ' + text)


# this function is copies from io.tools.py and adapted to also work on analogsignalarrays instead of analogsignals
def populate_RecordingChannel(bl, remove_from_annotation=True):
    """
    When a Block is
    Block>Segment>AnalogSIgnal
    this function auto create all RecordingChannel following these rules:
      * when 'channel_index ' is in AnalogSIgnal the corresponding
        RecordingChannel is created.
      * 'channel_index ' is then set to None if remove_from_annotation
      * only one RecordingChannelGroup is created

    It is a utility at the end of creating a Block for IO.

    Usage:
    >>> populate_RecordingChannel(a_block)
    """
    recordingchannels = {}
    for seg in bl.segments:
        # neo.io.tools version:
        # for anasig in seg.analogsignals:
        for anasig in seg.analogsignalarrays:
            if getattr(anasig, 'channel_index', None) is not None:
                ind = int(anasig.channel_index)
                if ind not in recordingchannels:
                    recordingchannels[ind] = RecordingChannel(index=ind)
                    if 'channel_name' in anasig.annotations:
                        channel_name = anasig.annotations['channel_name']
                        recordingchannels[ind].name = channel_name
                        if remove_from_annotation:
                            anasig.annotations.pop('channel_name')
                recordingchannels[ind].analogsignals.append(anasig)
                anasig.recordingchannel = recordingchannels[ind]
                if remove_from_annotation:
                    anasig.channel_index = None

    indexes = np.sort(list(recordingchannels.keys())).astype('i')
    names = np.array([recordingchannels[idx].name for idx in indexes],
                     dtype='S')
    rcg = RecordingChannelGroup(name='all channels',
                                channel_indexes=indexes,
                                channel_names=names)
    bl.recordingchannelgroups.append(rcg)
    for ind in indexes:
        # many to many relationship
        rcg.recordingchannels.append(recordingchannels[ind])
        recordingchannels[ind].recordingchannelgroups.append(rcg)