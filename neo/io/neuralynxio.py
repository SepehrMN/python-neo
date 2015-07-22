# -*- coding: utf-8 -*-
"""
Class for reading data from Neuralynx files.
This IO supports NCS, NEV and NSE file formats.

Depends on: numpy

Supported: Read

Author: ccanova, jsprenger
Adapted from the exampleIO of python-neo
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import logging
import struct
import sys
import os
import warnings
import copy
import re
import datetime

import numpy as np
import quantities as pq



from neo.io.baseio import BaseIO
from neo.core import (Block, Segment,
                      RecordingChannel, RecordingChannelGroup, AnalogSignalArray,
                      SpikeTrain, EventArray,Unit)
from neo.io import tools
from os import listdir
from os.path import isfile, join, getsize

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
    mode = 'file'

    # hardcoded parameters from manual
    nev_time_unit = pq.microsecond
    ncs_time_unit = pq.microsecond
    nse_time_unit = pq.microsecond




    def __init__(self, sessiondir=None, cachedir = None, print_diagnostic=False):
        """
        Arguments:
            sessiondir : the directory the files of the recording session are
                            collected. Default 'None'.
            print_diagnostic: indicates, whether information about the loading of
                            data is printed in terminal or not. Default 'False'.
            cachedir: the directory where metadata about the recording session is
                            read from and written to.

        """

        # TODO: Implement checksum and cache directory
        BaseIO.__init__(self)

        if sessiondir == None:
            raise ValueError('Must provide a directory containing data files of'
                                ' of one recording session.')

        # remove filename if specific file was passed
        if sessiondir.endswith('.ncs') \
            or sessiondir.endswith('.nev') \
            or sessiondir.endswith('.nse'):
            sessiondir = sessiondir[:sessiondir.rfind('/')]

        # remove / for consistent directory handling
        if sessiondir.endswith('/'):
                sessiondir = sessiondir.strip('/')

        # set general parameters of this IO
        self.sessiondir = sessiondir
        self._print_diagnostic = print_diagnostic
        self.associated = False
        self._associate(cachedir=cachedir)

        self._diagnostic_print('Initialized IO for session %s'%self.sessiondir)



    def read_block(self, lazy=False, cascade=True, t_starts=[None], t_stops=[None],
                    channel_list=[], units=[], analogsignals=True, events=False,
                    waveforms = False):
        """
        Reads data in a requested time window and returns block with single segment
        containing these data.

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
            channel_list : list of integers containing the IDs of the requested
                            to load. If [] all available channels will be loaded.
                            Default: [].
            units : list of integers containing the IDs of the requested units
                            to load. If [] all available units will be loaded.
                            Default: [].
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
                                   channel_list = [1,5,10], units = [1,2,3],
                                   events = True, waveforms = True)
        """
        # Create block
        bl = Block(file_origin=self.sessiondir)
        if not cascade:
            return bl

        # Checking Input
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

        sampling_rate = 1*pq.CompoundUnit('%i*Hz'%(self.parameters_ncs.values()[0]['sampling_rate']))
        # adapting t_starts and t_stops to known gap times
        for gap in self.parameters_global['gaps']:
            gap=gap[0]
            for e in range(len(t_starts)):
                t1,t2 = t_starts[e], t_stops[e]
                gap_start = (gap[1] - self.parameters_global['t_start']) *self.ncs_time_unit
                gap_stop = (gap[2] - self.parameters_global['t_start']) *self.ncs_time_unit
                if ((t1==t2==None)
                        or (t1==None and t2!=None and t2.rescale(self.ncs_time_unit)>gap_stop)
                        or (t2==None and t1!=None and t1.rescale(self.ncs_time_unit)<gap_stop)
                        or (t1!=None and t2!=None and t1.rescale(self.ncs_time_unit)<gap_start and t2.rescale(self.ncs_time_unit)>gap_stop)):
                    #adapting first time segment
                    t_stops[e]=gap_start
                    #inserting second time segment
                    t_starts.insert(e+1,gap_stop)
                    t_stops.insert(e+1,t2)


        #loading all channels if empty channel_list
        if channel_list == []:
            channel_list = self.parameters_ncs.keys()

        # adding a segment for each t_start, t_stop pair
        for t_start,t_stop in zip(t_starts,t_stops):
            seg = self.read_segment(lazy=lazy, cascade=cascade,
                                    t_start=t_start, t_stop=t_stop,
                                    channel_list=channel_list, units=units,
                                    analogsignals=analogsignals, events=events,
                                    waveforms=waveforms)
            bl.segments.append(seg)
        tools.populate_RecordingChannel(bl, remove_from_annotation=False)

        # This create rc and RCG for attaching Units
        rcg0 = bl.recordingchannelgroups[0]
        def find_rc(chan):
            for rc in rcg0.recordingchannels:
                if rc.index==chan:
                    return rc
        for seg in bl.segments:
            for st in seg.spiketrains:
                chan = st.annotations['channel_index']
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

        return bl


    def read_segment(self,lazy=False, cascade=True, t_start=None, t_stop=None,
                        channel_list=[], units=[], analogsignals=True,
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
            channel_list : list of integers containing the IDs of the requested
                            to load. If [] all available channels will be loaded.
                            Default: [].
            units : list of integers containing the IDs of the requested units
                            to load. If [] all available units will be loaded.
                            Default: [].
            events : Loading events. If True all available events in the given
                            time window will be read. Default: False.
            waveforms : Load waveform for spikes in the requested time
                            window. Default: False.


        Returns:
            Segment object containing neo objects, which contain the data.
        """

        # input check
        #loading all channels if empty channel_list
        if channel_list == []:
            channel_list = self.parameters_ncs.keys()
        elif [v for v in channel_list if v in self.parameters_ncs.keys()]== []:
            # warn if non of the requested channels are present in this session
            warnings.warn('Requested channels %s are not present in session '
                 '(contains only %s)'%(channel_list,self.parameters_ncs.keys()))
            channel_list = []


        seg = Segment(file_origin=self.filename)
        if not cascade:
            return seg

        # Reading NCS Files #
        # selecting ncs files to load based on channel_list requested
        if analogsignals:
            for channel_id in channel_list:
                if channel_id in self.parameters_ncs:
                    file_ncs = self.parameters_ncs[channel_id]['filename']
                    self.read_ncs(file_ncs, seg, lazy, cascade, t_start=t_start, t_stop = t_stop)
                else:
                    self._diagnostic_print('Can not load ncs of channel %i. No corresponding ncs file present.'%(channel_id))

        # Reading NEV Files (Events)#
        # reading all files available
        if events:
            for filename_nev in self.nev_avail:
                self.read_nev(filename_nev, seg, lazy, cascade, t_start = t_start, t_stop = t_stop)

        # Reading NSE Files (Spikes)#
        # reading all nse files available  #TODO: Load only spikes of requested channels
        for filename_nse in self.nse_avail:
            self.read_nse(filename_nse, seg, lazy, cascade, t_start = t_start, t_stop = t_stop, waveforms = waveforms)

        return seg



    # TODO: Option to load ncs based on channel_id instead of filename? Option to load ncs without providing segment?
    def read_ncs(self, filename_ncs, seg, lazy=False, cascade=True, t_start = None, t_stop = None):
        '''
        Reading a single .ncs file from the associated Neuralynx recording session.

        Arguments:
            filename_ncs : Name of the .ncs file to be loaded.
            seg : Neo Segment, to which the AnalogSignalArray containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            AnalogSignalArray. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time (quantity) that the AnalogSignalArray begins.
                            Default None.
            t_stop : time (quantity) that the AnalogSignalArray ends.
                            Default None.

        Returns:
            None

        Usage:
            TODO
        '''

        # checking format of filename and correcting if necessary
        if filename_ncs[-4:] != '.ncs':
            filename_ncs = filename_ncs + '.ncs'
        if '/' in filename_ncs:
            filename_ncs = filename_ncs.split('/')[-1]

        '''
        Extracting the channel id from prescan (association) of ncs files with
        this recording session
        '''


        chid = self.get_channel_id_by_file_name(filename_ncs)
        if chid == None:
            raise ValueError('NeuralynxIO is attempting to read a file '
                            'not associated to this session (%s).'%(filename_ncs))

        if not cascade:
            return



        #read data
        header_time_data = self.__mmap_ncs_packet_headers(filename_ncs)[0]

        data = self.__mmap_ncs_data(filename_ncs)

        # ensure meaningful values for requested start and stop times
        # + rescaling minimal time to 0ms
        if t_start==None or t_start < (self.parameters_ncs[chid]['t_start'] - self.parameters_global['t_start'] ) * self.ncs_time_unit:
            t_start = (self.parameters_ncs[chid]['t_start'] - self.parameters_global['t_start']) * self.ncs_time_unit
        if t_stop==None or t_stop > (self.parameters_ncs[chid]['t_stop'] - self.parameters_global['t_start']) *self.ncs_time_unit:
            t_stop= (self.parameters_ncs[chid]['t_stop']  - self.parameters_global['t_start']) *self.ncs_time_unit

        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s).'%(t_start,t_stop))

        unit = pq.dimensionless # default value
        if lazy:
            sig = []
            p_id_start = 0
        else:

            tstamps = (header_time_data - self.parameters_global['t_start']) * self.ncs_time_unit

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
            p_id_stop = min(first_stop + \
                            [gap_id[0] for gap_id in self.parameters_ncs[chid]['gaps'] if gap_id[0]>p_id_start] + \
                            [len(data)])

            # construct signal in valid packet range
            sig = np.array(data[p_id_start:p_id_stop+1],dtype=float)
            sig = sig.reshape(len(sig)*len(sig[0]))



            # Not guaranteed to be present in the header!
            if 'ADBitVolts' in self.parameters_ncs[chid]:
                sig *= self.parameters_ncs[chid]['ADBitVolts'] #Strong Assumption! Check Validity!
                unit = pq.V
            ################TODO: Check transformation of recording signal into physical signal!

        #defining sampling rate for rescaling purposes
        sampling_rate = 1*pq.CompoundUnit('%i*Hz'%(self.parameters_ncs[chid]['sampling_rate']))
        #creating neo AnalogSignalArray containing data
        anasig = AnalogSignalArray(signal = pq.Quantity(sig,unit, copy = False),
                                                    sampling_rate = sampling_rate,
                                                    # rescaling t_start to sampling time units
                                                    t_start = ((header_time_data[p_id_start] - self.parameters_global['t_start']) * self.ncs_time_unit).rescale(1/sampling_rate),
                                                    name = 'channel_%i'%(chid),
                                                    channel_index = chid)

        # removing protruding parts of first and last data packet
        if anasig.t_start < t_start.rescale(anasig.t_start.units):
            anasig = anasig.time_slice(t_start.rescale(anasig.t_start.units),None)

        if anasig.t_stop > t_stop.rescale(anasig.t_start.units):
            anasig = anasig.time_slice(None,t_stop.rescale(anasig.t_start.units))

        anasig.annotations = self.parameters_ncs[chid]

        seg.analogsignalarrays.append(anasig)


    def read_nev(self, filename_nev, seg, lazy=False, cascade=True, t_start=None, t_stop=None,
                     channel_list=[]):
        '''
        Reads associated nev file and attaches its content as eventarray to
        provided neo segment.

        Arguments:
            filename_nev : Name of the .nev file to be loaded.
            seg : Neo Segment, to which the EventArray containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            EventArray. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time (quantity) that the EventArray begins. Default None.
            t_stop : time (quantity) that the EventArray ends. Default None.

        Returns:
            None

        Usage:
            TODO
        '''


        # ensure meaningful values for requested start and stop times
        # + rescaling minimal time to 0ms
        if t_start==None or t_start < (self.parameters_nev['t_start'] - self.parameters_global['t_start'] ) * self.nev_time_unit:
            t_start = (self.parameters_nev['t_start'] - self.parameters_global['t_start']) * self.nev_time_unit
        if t_stop==None or t_stop > (self.parameters_nev['t_stop'] - self.parameters_global['t_start']) *self.nev_time_unit:
            t_stop= (self.parameters_nev['t_stop']  - self.parameters_global['t_start']) *self.nev_time_unit

        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s).'%(t_start,t_stop))


        data = self.__mmap_nev_file(filename_nev)

        for marker_i, name_i in self.parameters_nev['digital_markers'].iteritems():
            # Extract all time stamps of digital markers and rescaling time
            marker_times = np.array([event[3]-self.parameters_global['t_start'] for event in data if event[4]==marker_i])

            if self.parameters_global['spike_offset'] == None: offset=0
            else: offset=self.parameters_global['spike_offset']

            #only consider Events in the requested time window (t_start, t_stop) TODO!!!!!!
            marker_times = marker_times[((marker_times-offset) > t_start.rescale(self.nev_time_unit).magnitude) &
                                        ((marker_times-offset) > t_stop.rescale(self.nev_time_unit).magnitude)]

            ev = EventArray(times=pq.Quantity(marker_times-offset, units=self.nev_time_unit, dtype="int"),
                                labels= name_i,
                                name="Digital Marker " + str(marker_i),
                                file_origin=filename_nev,
                                marker_id=marker_i,
                                digital_marker=True,
                                analog_marker=False,
                                analog_channel=0)

            seg.eventarrays.append(ev)

    def read_nse(self, filename_nse, seg, lazy=False, cascade=True, t_start=None, t_stop=None,
                     waveforms = False):
        '''
        Reads nse file and attaches content as spike train to provided neo segment.

        Arguments:
            filename_nse : Name of the .nse file to be loaded.
            seg : Neo Segment, to which the Spiketrain containing the data
                            will be attached.
            lazy : Postpone actual reading of the data. Instead provide a dummy
                            SpikeTrain. Default 'False'.
            cascade : Not used in this context. Default: 'True'.
            t_start : time (quantity) that the SpikeTrain begins. Default None.
            t_stop : time (quantity) that the SpikeTrain ends. Default None.
            waveforms : Load the waveform (up to 32 data points) for each
                            spike time. Default: False

        Returns:
            None

        Usage:
            TODO
        '''

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
        # + rescaling minimal time to 0ms
        if t_start==None or t_start < (self.parameters_nse[chid]['t_start'] - self.parameters_global['t_start'] ) * self.nse_time_unit:
            t_start = (self.parameters_nse[chid]['t_first'] - self.parameters_global['t_start']) * self.nse_time_unit

        # using t_stop of ncs, because nse does not contain reliable stopt time.
        if chid in self.parameters_ncs:
            if t_stop==None or t_stop > (self.parameters_ncs[chid]['t_stop'] - self.parameters_global['t_start']) *self.nse_time_unit:
                t_stop= (self.parameters_ncs[chid]['t_stop']  - self.parameters_global['t_start']) *self.nse_time_unit
        else:
            if t_stop==None:
                t_stop= (sys.maxsize) *self.nse_time_unit

        if t_start >= t_stop:
            raise ValueError('Requested start time (%s) is later than / equal to stop time (%s).'%(t_start,t_stop))


        # reading data
        [timestamps, channel_ids, cell_numbers, features, data_points] = self.__mmap_nse_packets(filename_nse)


        # collecting spike times for each individual unit (assuming unit numbers
        # start at 0 and go to #(number of units)
        for unit_i in range(self.parameters_nse[chid]['cell_count']):

            if not lazy:
                # Extract all time stamps of that neuron on that electrode
                spike_times = np.array([time[0] for time in timestamps
                                        if time[1][4][1]==unit_i])
                spikes = pq.Quantity((spike_times - self.parameters_global['t_start']),
                                        units=self.nse_time_unit)
            else:
                spikes = pq.Quantity([], units=self.nse_time_unit)

            # Create SpikeTrain object
            st = SpikeTrain(times=spikes,
                                dtype='int',
                                t_start=t_start,
                                t_stop=t_stop,
                                sampling_rate=self.parameters_ncs.values()[0]['sampling_rate'],
                                name= "Channel %i, Unit %i"%(chid, unit_i),
                                file_origin=filename_nse,
                                unit_id=unit_i,
                                channel_id=chid)

            if waveforms and not lazy:
                # Collect all waveforms of the specific unit
                # For computational reasons: no units, no time axis
                st.waveforms = np.array([data_points[t,:] for t in range(len(timestamps)) if cell_numbers[t]==unit_i])

            # annotation of spiketrains?
            seg.spiketrains.append(st)


############# private routines #################################################



    def _associate(self, cachedir = None):
        """
        Associates the object with a specified Neuralynx session, i.e., a
        combination of a .nse, .nev and .ncs files. The meta data is read into the
        object for future reference.

        Arguments:
        Returns:
            -
        """

        # If already associated, disassociate first
        if self.associated:
            raise IOError("Trying to associate an already associated \
                NeuralynxIO object.")

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
        self.sessionfiles = [ f for f in listdir(self.sessiondir) if isfile(join(self.sessiondir,f)) ]


        self.ncs_avail = []
        self.nse_avail = []
        self.nev_avail = []
        self.ntt_avail = []

        # check if there are any changes of the data files -> new data check run
        check_files = True
        if cachedir != None:
            cachefile = cachedir + '/' + self.sessiondir.split('/')[-1] + '/hashkeys'
            if not os.path.exists(cachedir + '/' + self.sessiondir.split('/')[-1]):
                os.makedirs(cachedir + '/' + self.sessiondir.split('/')[-1])

            if os.path.exists(cachefile):
                hashes_read = pickle.load(open(cachefile, 'rb') )
            else: hashes_read = {}

            hashes_calc = {f:self.hashfile(open(self.sessiondir + '/' + f, 'rb'), hashlib.sha256()) for f in self.sessionfiles}
            if all([f in hashes_calc and f in hashes_read and hashes_calc[f] == hashes_read[f] for f in self.sessionfiles]):
                check_files = False
                self._diagnostic_print('Using cached metadata from earlier analysis run in file %s. Skip file checks.'%cachefile)

                # loading saved parameters
                parameterfile = cachedir + '/' + self.sessiondir.split('/')[-1] + '/parameters.cache'
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
                self.__read_ncs_text_header(ncs_file,channel_id)

                # Check for invalid starting times of data packets in ncs file
                self.__ncs_invalid_first_sample_check(filehandle)

                # Check ncs file for gaps
                self.__ncs_gap_check(filehandle)


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
                self.__read_nse_text_header(nse_file,channel_id)


            #=======================================================================
            # # Scan NEV files
            #=======================================================================

            self._diagnostic_print('\nDetected %i .nev file(s).'%(len(self.nev_avail)))

            for nev_file in self.nev_avail:
                # Loading individual NEV file and extracting parameters
                self._diagnostic_print('Scanning ' + nev_file + '.')

                # Reading file
                filehandle = self.__mmap_nev_file(nev_file)

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
                self.__read_nev_text_header(nev_file)



            #=======================================================================
            # # Scan NTT files
            #=======================================================================

            self._diagnostic_print('\nDetected %i .ntt file(s).'%(len(self.ntt_avail)))

            for ntt_file in self.ntt_avail:
                # Loading individual NTT file and extracting parameters
                self._diagnostic_print('Scanning ' + ntt_file + '.')

                # Reading file
                filehandle = self.__mmap_ntt_file(ntt_file)

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

            #=======================================================================
            # # Check consistency across files
            #=======================================================================

            # TODO: Implement also ntt files in this part
            # check starting times of .ncs files
            if len(np.unique([i['t_start'] for i in self.parameters_ncs.values()])) > 1:
                raise ValueError('NCS files do not start at same time point.')


            # check recoding_opened times (from txt header) for different files
            # This is performed file type wise as there can be opening differences of up to 15 seconds...
            # TODO: find out why this is the case (see above)
            nev_parameters = {None: self.parameters_nev} if self.parameters_nev else {}
            for parameter_collection in [self.parameters_ncs, self.parameters_nse, nev_parameters]:
                if any(np.abs(np.diff([i['recording_opened'] for i in parameter_collection.values()]))>datetime.timedelta(seconds=0.1)):
                    raise ValueError('NCS files were opened for recording with a delay greater than 0.1 second.')

                # check recoding_opened times (from txt header) for different ncs files
                if any(np.diff([i['recording_closed'] for i in parameter_collection.values() if i['recording_closed'] != None])>datetime.timedelta(seconds=0.1)):
                    raise ValueError('NCS files were closed after recording with a delay greater than 0.1 second.')

            nev_parameters = [self.parameters_nev] if self.parameters_nev else []
            parameter_collection = self.parameters_ncs.values() + self.parameters_nse.values() + nev_parameters
            self.parameters_global['recording_opened'] = min([i['recording_opened']for i in parameter_collection])
            self.parameters_global['recording_closed'] = max([i['recording_closed']for i in parameter_collection])


            self.parameters_global['t_start'] = 0
            self.parameters_global['event_offset'] = 0
            # check if also nev file starts at same time point
            if self.nev_avail!=[] and self.ncs_avail != {} and self.parameters_ncs.values()[0]['t_start'] != self.parameters_nev['Starting_Recording'][0]:
                warnings.warn('NCS and event of recording start are not the same!')

            # check if nse time is available and extract first time point as t_first
            if self.nse_avail != [] and \
                           self.parameters_nse.values()[0]['t_first'] != None:
               t_first = self.parameters_nse.values()[0]['t_first']
            else: t_first = np.inf #using inf, because None is handles as if neg. number

            #setting global time frame
            if self.nev_avail!=[]:
                self.parameters_global['t_start'] = min(self.parameters_ncs.values()[0]['t_start'],
                                                    self.parameters_nev['t_start'],
                                                    t_first)
                self.parameters_global['event_offset'] = self.parameters_nev['t_start'] \
                                                            - self.parameters_ncs.values()[0]['t_start']
            else:
                self.parameters_global['t_start'] = min(self.parameters_ncs.values()[0]['t_start'], t_first)
                self.parameters_global['event_offset'] = self.parameters_ncs.values()[0]['t_start']

            # Offset time of .nse file can not be determined for sure as there is no
            # time stamp of recording start in this file -> check by by comparison to .ncs
            self.parameters_global['spike_offset'] = None



            # checking gap consistency
            # across ncs files
            #check number of gaps detected
            if len(np.unique([len(i['gaps']) for i in self.parameters_ncs.values()])) != 1:
                raise ValueError('NCS files contain different numbers of gaps!')
            # check consistency of gaps across files and create global gap collection
            self.parameters_global['gaps'] = []
            for g in range(len(self.parameters_ncs.values()[0]['gaps'])):
                if len(np.unique([i['gaps'][g] for i in self.parameters_ncs.values()])) != 1:
                    raise ValueError('Gap number %i is not consistent across NCS files.'%(g))
                else:
                    self.parameters_global['gaps'].append(self.parameters_ncs.values()[0]['gaps'])


        # save results of association for future analysis
        if cachedir != None:
            pickle.dump( {'global': self.parameters_global,
                          'ncs': self.parameters_ncs,
                          'nev': self.parameters_nev,
                          'nse': self.parameters_nse,
                          'ntt': self.parameters_ntt},
                         open( cachedir + '/' + self.sessiondir.split('/')[-1] + '/parameters.cache', 'wb' ) )
            pickle.dump( hashes_calc, open(cachedir + '/' + self.sessiondir.split('/')[-1] + '/hashkeys', 'wb' ))

            # with open(cachedir + '/' + self.sessiondir.split('/')[-1] + '/hashkeys', 'w' ) as file:
            #     for name,hash in hashes_calc.iteritems():
            #         file.write(name + '\t' + hash + '\n')

        self.associated = True


#################### private routines #######################################


################# Memory Mapping Methods

    def __mmap_nse_packets(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        filesize = getsize(self.sessiondir + '/' + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + '/' + filename,
                            dtype='<u2', shape = ((filesize-16384)/2/56,56),
                            mode='r', offset=16384)

            # reconstructing original data
            timestamps = data[:,0] + data[:,1]*2**16 + data[:,2]*2*32 + data[:,3]*2*48 # first 4 ints -> timestamp in microsec
            channel_id = data[:,4] + data[:,5]*2**16
            cell_number = data[:,6] + data[:,7]*2**16
            features = [data[:,p] + data[:,p+1]*2**16 for p in range(8,23,2)] #this is inconsistent with the Neuraview output as this can be negative
            features = np.array(features,dtype='i4')

            data_points = data[:,24:56].astype('i2')
            del data
            return timestamps, channel_id, cell_number, features, data_points
        else: return None


    def __mmap_ncs_data(self,filename):
        """ Memory map of the Neuralynx .ncs file optimized for data extraction"""
        if getsize(self.sessiondir + '/' + filename) > 16384:
            data = np.memmap(self.sessiondir + '/' + filename, dtype=np.dtype(('i2',(522))),mode='r', offset=16384)
            #removing data packet headers and flattening data
            return data[:,10:]
        else: return None

    def __mmap_ncs_packet_headers(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        filesize = getsize(self.sessiondir + '/' + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + '/' + filename,
                            dtype='<u4', shape = ((filesize-16384)/4/261,261),
                            mode='r', offset=16384)

            timestamps = data[:,0] + (data[:,1] *2**32)
            header_u4 = data[:,2:5]
            del data
            return timestamps, header_u4
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

        if getsize(self.sessiondir + '/' + filename) > 16384:
            return np.memmap(self.sessiondir + '/' + filename,
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
        if getsize(self.sessiondir + '/' + filename) > 16384:
            return np.memmap(self.sessiondir + '/' + filename,
                                         dtype=nse_dtype, mode='r', offset=16384)
        else: return None


    def __mmap_ntt_packets(self,filename):
        """
        Memory map of the Neuralynx .ncs file optimized for extraction of data packet headers
        Reading standard dtype improves speed, but timestamps need to be reconstructed
        """
        # TODO: Check if 'CHANNEL' really never occurs in file (last entry of data packet
        filesize = getsize(self.sessiondir + '/' + filename) #in byte
        if filesize > 16384:
            data = np.memmap(self.sessiondir + '/' + filename,
                            dtype='<u2', shape = ((filesize-16384)/2/152,152),
                            mode='r', offset=16384)

            # reconstructing original data
            timestamps = data[:,0] + data[:,1]*2**16 + data[:,2]*2*32 + data[:,3]*2*48 # first 4 ints -> timestamp in microsec
            channel_id = data[:,4] + data[:,5]*2**16
            cell_number = data[:,6] + data[:,7]*2**16
            features = [data[:,p] + data[:,p+1]*2**16 for p in range(8,23,2)] #this is inconsistent with the Neuraview output as this can be negative
            features = np.array(features,dtype='i4')

            data_points = data[:,24:152].astype('i2').reshape((4,32))
            del data
            return timestamps, channel_id, cell_number, features, data_points
        else: return None


    #___________________________ header extraction __________________________

    def __read_ncs_text_header(self, filename_ncs, chid):
        # Reading main file header (plain text, 16kB)
        ncs_text_header = open(self.sessiondir + '/' + filename_ncs,'r').read(16384)
        #separating lines of header and ignoring last line (fill)
        ncs_text_header = ncs_text_header.split('\r\n')[:-1]

        # extracting filename and recording opening/closing time
        header_dict = self.__read_intro_txt_header(ncs_text_header)
        self.parameters_ncs[chid].update(header_dict)


        # minor parameters possibly saved in header
        ncs_minor_keys =  ['CheetahRev','AcqEntName','FileType','RecordSize',
                          'HardwareSubSystemName','HardwareSubSystemType',
                          'SamplingFrequency','ADMaxValue','ADBitVolts','NumADChannels',
                          'ADChannel','InputRange','InputInverted','DSPLowCutFilterEnabled',
                          'DspLowCutFrequency','DspLowCutNumTaps','DspLowCutFilterType',
                          'DSPHighCutFilterEnabled','DspHighCutFrequency','DspHighCutNumTaps',
                          'DspHighCutFilterType','DspDelayCompensation','DspFilterDelay_\xb5s']


        #extracting minor key values of header (only taking into account non-empty lines)
        for i, minor_entry in enumerate([text for text in ncs_text_header[4:] if text != '']):
            if minor_entry.split(' ')[0] in ['-' + ncs_minor_keys[i] for i in range(len(ncs_minor_keys))]:

                # determine data type of entry
                minor_value = minor_entry.split(' ')[1]
                if minor_value.isdigit():
                    minor_value = int(minor_value)
                else:
                    try:
                        minor_value = float(minor_value)
                    except:
                        pass

                # assign value of correct data type to ncs parameter dictionary
                self.parameters_ncs[chid][minor_entry.split(' ')[0][1:]] = minor_value

        self._diagnostic_print('Successfully decoded text header of ncs file (%s).'%(filename_ncs))



    def __read_nse_text_header(self,filename_nse,chid):
        # Reading main file header (plain text, 16kB)
        nse_text_header = open(self.sessiondir + '/' + filename_nse,'r').read(16384)
        #separating lines of header and ignoring last line (fill)
        nse_text_header = nse_text_header.split('\r\n')[:-1]

        # extracting filename and recording opening/closing time
        header_dict = self.__read_intro_txt_header(nse_text_header)
        self.parameters_nse[chid].update(header_dict)

    def __read_ntt_text_header(self,filename_ntt,chid):
        # Reading main file header (plain text, 16kB)
        ntt_text_header = open(self.sessiondir + '/' + filename_ntt,'r').read(16384)
        #separating lines of header and ignoring last line (fill)
        ntt_text_header = ntt_text_header.split('\r\n')[:-1]

        # extracting filename and recording opening/closing time
        header_dict = self.__read_intro_txt_header(ntt_text_header)
        self.parameters_ntt[chid].update(header_dict)

    def __read_nev_text_header(self,filename_nev):
        # Reading main file header (plain text, 16kB)
        nev_text_header = open(self.sessiondir + '/' + filename_nev,'r').read(16384)
        #separating lines of header and ignoring last line (fill)
        nev_text_header = nev_text_header.split('\r\n')[:-1]

        # extracting filename and recording opening/closing time
        header_dict = self.__read_intro_txt_header(nev_text_header)
        self.parameters_nev.update(header_dict)


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
            datetime_struct = re.compile('## Time (?P<mode>\S{6}) \(m/d/y\): (?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})  '
                                                        '\(h:m:s\.ms\) (?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})\.(?P<millisecond>\d{1,3})')

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
                    warnings.warn('Text header of file %s does not contain recording closed time. File was not closed properly.'%output['recording_file_name'])
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
        #calculating corresponding time stamp of first sample, that was not
        #recorded any more
        #       time of first sample in last packet + (number of sample per packet * conversion factor (time are recorded in ms) / sampling rate
        t_stop = timestamps[-1] + ((header_u4[-1][2]) * (1/self.ncs_time_unit.rescale(pq.s)).magnitude / header_u4[-1][1])

        if channel_id in self.parameters_ncs:
            raise ValueError('Detected multiple ncs files for channel_id %i.'%(channel_id))
        else:
            self.parameters_ncs[channel_id] = { 'filename':filename,
                                                't_start': t_start,
                                                't_stop': t_stop,
                                                'sampling_rate': sr,
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
                                                't_first': t_first,
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
                                                't_first': t_first,
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
        if 'Starting_Recording' in self.parameters_nev:
            raise ValueError('Trying to read second nev file. Only one can be handled.')
        self.parameters_nev['Starting_Recording'] = []
        for event in filehandle:
            if event[4] == 11: # meaning recording start
                self.parameters_nev['Starting_Recording'].append(event[3])

        if len(self.parameters_nev['Starting_Recording']) < 1:
            raise ValueError('No Event "Starting_Recording" detected in %s'%(filename))

        self.parameters_nev['t_start'] = min(self.parameters_nev['Starting_Recording'])
        self.parameters_nev['t_stop'] = filehandle[-1][3] # t_stop = time stamp of last event in file (assuming chronological order)



        # extract all occurring event marker ids
        event_type_collection = {}
        for event in filehandle:
            event_type_collection[event[4]] = event[10]
        self.parameters_nev['digital_markers'] = copy.deepcopy(event_type_collection)  # entries (marker_id,name)


        # TODO: Extract other important Recording events. But which ones?



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
        # this is a not safe assumption, that the first two data packets have correct time stamps
        delta_t = timestamps[1] - timestamps[0]

        # valid samples of first data packet
        temp_valid_samples = header_u4[0,2]

        # unit test
        # time difference between packets corresponds to number of recorded samples
        # 10**6 due to unit conversion microsec -> sec
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

        assert all(cell_numbers == cell_numbers[0])

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

        # TODO: Not yet implemented. What should be tested?
        pass
#        self._diagnostic_print('NEV file check successful.')


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

        assert all(cell_numbers == cell_numbers[0])

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
        packet_checks = (valid_samples / (self.ncs_time_unit.rescale(pq.s).magnitude * sampling_rate)) == data_packet_offsets
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
        Checking parameters of NCS and NSE Files for given filename and return channel_id if result is consistent
        :param filename:
        :return:
        """
        channel_ids = []
        channel_ids += [k for k in self.parameters_ncs if self.parameters_ncs[k]['filename'] == filename]
        channel_ids += [k for k in self.parameters_nse if self.parameters_nse[k]['filename'] == filename]
        if len(np.unique(np.asarray(channel_ids))) == 1:
            return  channel_ids[0]
        elif len(channel_ids) > 1:
            raise ValueError('Ambiguous channel ids detected. Filename %s is associated to different channels of '
            'NCS and NSE %s'%(filename,channel_ids))
        else: # if filename was not detected
            return None


    def hashfile(self,afile, hasher, blocksize=65536):
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
        return hasher.digest()




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