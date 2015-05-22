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
import warnings
import copy

import numpy as np
import quantities as pq



from neo.io.baseio import BaseIO
from neo.core import (Block, Segment,
                      RecordingChannel, RecordingChannelGroup, AnalogSignalArray,
                      SpikeTrain, EventArray)
from neo.io import tools
from os import listdir
from os.path import isfile, join, getsize




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




    def __init__(self, sessiondir=None, print_diagnostic=False, check_files=True):
        """
        Arguments:
            sessiondir : the directory the files of the recording session are
                            collected. Default 'None'.
            print_diagnostic: indicates, whether information about the loading of
                            data is printed in terminal or not. Default 'False'.
            check_files: check associated files on consistency. This is highly
                            recommended to ensure correct operation. Default 'True'

        """
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
        self._associate(check_files = check_files)

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

        channel_id = [k for k in self.parameters_ncs.keys() if self.parameters_ncs[k]['filename'] == filename_ncs]
        if len(channel_id) == 1:
            chid = channel_id[0]
        else:
            raise ValueError('NeuralynxIO is attempting to read a file '
                            'not associated to this session (%s).'%(filename_ncs))




        # Reading main file header (plain text, 16kB)
        ncs_text_header = open(self.sessiondir + '/' + filename_ncs,'r').read(16384)
        #separating lines of header and ignoring last line (fill)
        ncs_text_header = ncs_text_header.split('\r\n')[:-1]
        try:
            # checking title of header
            if ncs_text_header[0] != '######## Neuralynx Data File Header':
                raise TypeError('NCS file has unkown Neuralynx header title')

            # extracting major entries in header
            if  ncs_text_header[1].startswith('## File Name ') \
                and ncs_text_header[2].startswith('## Time Opened (m/d/y): ') \
                and '(h:m:s.ms)' in ncs_text_header[2] \
                and ncs_text_header[3].startswith('## Time Closed (m/d/y): ') \
                and '(h:m:s.ms)' in ncs_text_header[3]:


                self.parameters_ncs[chid]['Original_File_Name'] =  ncs_text_header[1].replace('## File Name ','')
                # Separating 'Time Opened/Closed' into date and time information # TODO: extract individual values for year/month/day/hour/minute/second/microsec
                self.parameters_ncs[chid]['Date_Opened_(m/d/y)'] = ncs_text_header[2].replace('## Time Opened (m/d/y): ','').split('(h:m:s.ms)')[0]
                self.parameters_ncs[chid]['Time_Opened_(h:m:s.ms)'] = ncs_text_header[2].replace('## Time Opened (m/d/y): ','').split('(h:m:s.ms)')[1]
                self.parameters_ncs[chid]['Date_Closed_(m/d/y)'] = ncs_text_header[3].replace('## Time Closed (m/d/y): ','').split('(h:m:s.ms)')[0]
                self.parameters_ncs[chid]['Time_Closed_(h:m:s.ms)'] = ncs_text_header[3].replace('## Time Closed (m/d/y): ','').split('(h:m:s.ms)')[1]
            else:
                raise TypeError('NCS file has unknown major parameters in header')

            # minor parameters posssibly saved in header
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

        except TypeError:
            warnings.warn('WARNING: NeuralynxIO is unable to extract data from text header! \
                            Continue with loading data.')

        if not cascade:
            return



        #read data
        header_time_data = self.__mmap_ncs_time_stamps(filename_ncs)

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

            #find data paket to start with signal construction
            starts = np.where(tstamps<=t_start)[0]
            if len(starts) == 0:
                self._diagnostic_print('Requested AnalogSignalArray not present in this time interval.')
                return
            else:
                #first paket to be included into signal
                p_id_start = starts[-1]
            #find data paket where signal ends (due to gap or t_stop)
            stops = np.where(tstamps>=t_stop)[0]
            if len(stops) !=0:
                first_stop = [stops[0]]
            else: first_stop = []

            # last paket to be included in signal
            p_id_stop = min(first_stop + \
                            [gap_id[0] for gap_id in self.parameters_ncs[chid]['gaps']] + \
                            [len(data)])

            # construct signal in valid paket range
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

        # removing protruding parts of first and last data paket
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
        channel_id = [k for k in self.parameters_nse.keys() if self.parameters_nse[k]['filename'] == filename_nse]
        if len(channel_id) == 1:
            channel_id = channel_id[0]
        else:
            #if nse file is empty it is not listed in self.parameters_nse, but
            # in self.nse_avail
            if filename_nse  in self.nse_avail:
                warnings.warn('NeuralynxIO is attempting to read an empty '
                            '(not associated) nse file (%s).'
                            'Not loading nse file.'%(filename_nse))
                return
            else:
                raise ValueError('NeuralynxIO is attempting to read a file '
                          'not associated to this session (%s).'%(filename_nse))

        # reading data
        data = self.__mmap_nse_file(filename_nse)

        # collecting spike times for each individual unit (assuming unit numbers
        # start at 0 and go to #(number of units)
        for unit_i in range(self.parameters_nse[channel_id]['cell_count']):

            if not lazy:
                # Extract all time stamps of that neuron on that electrode
                spike_times = np.array([time[0] for time in data
                                        if time[1][4][1]==unit_i])
                spikes = pq.Quantity((spike_times - self.parameters_global['t_start']),
                                        units=self.nse_time_unit)
            else:
                spikes = pq.Quantity([], units=self.nse_time_unit)

            # Create SpikeTrain object
            st = neo.SpikeTrain(times=spikes,
                                dtype='int',
                                t_start=t_start,
                                t_stop=t_stop,
                                sampling_rate=self.parameters_ncs.values()[0]['sampling_rate'],
                                name= "Channel %i, Unit %i"%(channel_id, unit_i),
                                file_origin=filehame_nse,
                                unit_id=unit_i,
                                channel_id=channel_i)

            if waveforms and not lazy:
                # Collect all waveforms of the specific unit
                # For computational reasons: no units, no time axis
                st.waveforms = np.array([time[4][0] for time in data if time[1][4][1]==unit_i])

            # annotation of spiketrains?
            seg.spiketrains.append(st)


############# private routines #################################################



    def _associate(self, check_files=True):
        """
        Associates the object with a specified Neuralynx session, i.e., a
        combination of a .nse, .nev and .ncs files. The meta data is read into the
        object for future reference.

        Arguments:
            check_files: check associated files on consistency. This is highly
                            recommended to ensure correct operation.
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
        # List of parameter dictionaries for all potential .nsX files 0..9
        self.parameters_ncs = {}
        self.parameters_nev = {}

        # Scanning session directory for recorded files
        self.sessionfiles = [ f for f in listdir(self.sessiondir) if isfile(join(self.sessiondir,f)) ]

        #=======================================================================
        # # Scan NCS files
        #=======================================================================

        self.ncs_avail = []
        self.nse_avail = []
        self.nev_avail = []

        for filename in self.sessionfiles:
            # Extracting only continuous signal files (.ncs)
            if filename[-4:] == '.ncs':
                self.ncs_avail.append(filename)

        self._diagnostic_print('\nDetected %i .ncs files.'%(len(self.ncs_avail)))



        for ncs_file in self.ncs_avail:
            # Loading individual NCS file and extracting parameters
            self._diagnostic_print("Scanning " + ncs_file + ".")

            # Reading file headers
            filehandle = self.__mmap_ncs_paket_headers(ncs_file)

            if check_files:
                # Checking consistency of ncs file
                self.__ncs_check(filehandle)

            # Reading header information and store them in parameters_ncs
            self.__read_ncs_header(filehandle, ncs_file)

            # Check for invalid starting times of data pakets in ncs file
            self.__ncs_invalid_first_sample_check(filehandle)

            # Check ncs file for gaps
            self.__ncs_gap_check(filehandle)


        #=======================================================================
        # # Scan NSE files
        #=======================================================================

        for filename in self.sessionfiles:
            # Extracting only single electrode spikes recording files (.nse)
            if filename[-4:] == '.nse':
                self.nse_avail.append(filename)

        self._diagnostic_print('\nDetected %i .nse files.'%(len(self.nse_avail)))

        for nse_file in self.nse_avail:
            # Loading individual NSE file and extracting parameters
            self._diagnostic_print('Scanning ' + nse_file + '.')

            # Reading file
            filehandle = self.__mmap_nse_file(nse_file)

            if check_files:
                # Checking consistency of nse file
                self.__nse_check(filehandle)

            # Reading header information and store them in parameters_nse
            self.__read_nse_header(filehandle, nse_file)


        #=======================================================================
        # # Scan NEV files
        #=======================================================================


        for filename in self.sessionfiles:
            # Extracting only single electrode spikes recording files (.nev)
            if filename[-4:] == '.nev':
                self.nev_avail.append(filename)

        self._diagnostic_print('\nDetected %i .nev files.'%(len(self.nev_avail)))

        for nev_file in self.nev_avail:
            # Loading individual NEV file and extracting parameters
            self._diagnostic_print('Scanning ' + nev_file + '.')

            # Reading file
            filehandle = self.__mmap_nev_file(nev_file)

            if check_files:
                # Checking consistency of nev file
                self.__nev_check(filehandle)

            # Reading header information and store them in parameters_nev
            self.__read_nev_header(filehandle, nev_file)

        #=======================================================================
        # # Check consistency across files
        #=======================================================================

        # check starting times of .ncs files
        if len(np.unique([i['t_start'] for i in self.parameters_ncs.values()])) != 1:
            raise ValueError('NCS files do not start at same time point.')

        self.parameters_global = {'t_start':0,'event_offset':0}
        # check if also nev file starts at same time point
        if self.parameters_ncs.values()[0]['t_start'] != self.parameters_nev['Starting_Recording'][0]:
            warnings.warn('NCS and event of recording start are not the same!')

        # check if nse time is available and extract first time point as t_first
        if self.nse_avail != [] and \
                       self.parameters_nse.values()[0]['t_first'] != None:
           t_first = self.parameters_nse.values()[0]['t_first']
        else: t_first = np.inf #using inf, because None is handles as if neg. number

        #setting global time frame
        self.parameters_global['t_start'] = min(self.parameters_ncs.values()[0]['t_start'],
                                            self.parameters_nev['Starting_Recording'][0],
                                            t_first)
        self.parameters_global['event_offset'] = self.parameters_nev['Starting_Recording'][0] \
                                                    - self.parameters_ncs.values()[0]['t_start']

        # Offset time of .nse file can not be determined for sure as there is no
        # time stamp of recording start in this file -> check by by comparison to .ncs
        self.parameters_global['spike_offset'] = None



        # checking gap consistency
        # across ncs files
        #check number of gaps detected
        if len(np.unique([len(i['gaps']) for i in self.parameters_ncs.values()])) != 1:
            raise ValueError('NCS files contain different numbers of gaps!')
        # check consistency of gaps across files
        for g in range(len(self.parameters_ncs.values()[0]['gaps'])):
            if len(np.unique([i['gaps'][g] for i in self.parameters_ncs.values()])) != 1:
                raise ValueError('Gap number %i is not consistent across NCS files.'%(g))










#################### private routines #######################################


################# Memory Mapping Methods

    def __mmap_nse_file(self, filename):
        """ Memory map the Neuralynx .nse file """
        nse_dtype = np.dtype([
            ('timestamp', '<u8'),
            ('sc_number', '<u4'),
            ('cell_number', '<u4'),
            ('params', '<u4',   (8,)),
            ('data', '<i2', (32, 1)),
        ])
        if getsize(self.sessiondir + '/' + filename) > 16384:
            return np.memmap(self.sessiondir + '/' + filename, dtype=nse_dtype, mode='r', offset=16384)
        else:
            return None

    def __mmap_nse_time_stamps(self, filename):
        """ Memory map the Neuralynx .nse file """
        nse_dtype = np.dtype([
            ('timestamp', '<u8'),
            ('rest', 'V102'),
            ('channel','<i2')])
        if getsize(self.sessiondir + '/' + filename) > 16384:
            data = np.memmap(self.sessiondir + '/' + filename, dtype=nse_dtype, mode='r', offset=16384)
            return copy.deepcopy(np.array([[i[0],i[2]] for i in data]))
        else:
            return None


    #different methods for reading different parts of ncs files in efficient ways
    def __mmap_ncs_file(self, filename):
        """ Memory map the Neuralynx .ncs file """
        ncs_dtype = np.dtype([
            ('timestamp', '<u8'),
            ('channel_number', '<u4'),
            ('sample_freq', '<u4'),
            ('valid_samples', '<u4'),
            ('samples', '<i2', (512)),
        ])
        return np.memmap(self.sessiondir + '/' + filename, dtype=ncs_dtype, mode='r', offset=16384)

    def __mmap_ncs_data(self,filename):
        """ Memory map of the Neuralynx .ncs file optimized for data extraction"""

        data = np.memmap(self.sessiondir + '/' + filename, dtype=np.dtype(('i2',(522))),mode='r', offset=16384)
        #removing data paket headers and flattening data
        return data[:,10:]

    def __mmap_ncs_time_stamps(self,filename):
        """ Memory map of the Neuralynx .ncs file optimized for extraction of time stamps of data pakets"""
        data = np.memmap(self.sessiondir + '/' + filename, dtype=np.dtype([('timestamp','<u8'),('rest','V%s'%(512*2+12))]),mode='r', offset=16384)
        return copy.deepcopy(np.array([i[0] for i in data],dtype=np.dtype('u8')))

    def __mmap_ncs_paket_headers(self,filename):
        """ Memory map of the Neuralynx .ncs file optimized for extraction of data paket headers"""
        data = np.memmap(self.sessiondir + '/' + filename,
                                dtype=np.dtype([('timestamp','<u8'),
                                                ('channel_number', '<u4'),
                                                ('sample_freq', '<u4'),
                                                ('valid_samples', '<u4'),
                                                ('rest','V%s'%(512*2))]),
                                mode='r', offset=16384)
        return copy.deepcopy(np.array([np.array([i[0],i[1],i[2],i[3]])
                                                        for i in data]))

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
        return np.memmap(self.sessiondir + '/' + filename,
                                     dtype=nev_dtype, mode='r', offset=16384)


    def __mmap_ntt_file(self, filename):
        """ Memory map the Neuralynx .nse file """
        nse_dtype = np.dtype([
            ('timestamp', '<u8'),
            ('sc_number', '<u4'),
            ('cell_number', '<u4'),
            ('params', '<u4',   (8,)),
            ('data', '<i2', (32, 4)),
        ])
        return np.memmap(self.sessiondir + '/' + filename,
                                     dtype=nse_dtype, mode='r', offset=16384)




    #___________________________ header extraction __________________________

    def __read_ncs_header(self, filehandle, filename):
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

        t_start = filehandle[0][0] # in microseconds
        #calculating corresponding time stamp of first sample, that was not
        #recorded any more
        #       time of first sample in last paket + (number of sample per paket *
        t_stop = filehandle[-1][0] + ((filehandle[-1][3]) *
                 # conversion factor (time are recorded in ms)
                 (1/self.ncs_time_unit.rescale(pq.s)).magnitude /
                 filehandle[-1][2]) # sampling rate
        channel_id = filehandle[0][1]
        sr = filehandle[0][2] # in Hz

        if channel_id in self.parameters_ncs:
            raise ValueError('Detected multiple files for channel_id %i.'%(channel_id))
        else:
            self.parameters_ncs[channel_id] = { 'filename':filename,
                                                't_start': t_start,
                                                't_stop': t_stop,
                                                'sampling_rate': sr,
                                                'gaps': []}

            return {channel_id: self.parameters_ncs[channel_id]}


    def __read_nse_header(self, filehandle, filename):
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

        # TODO: Extend extracted information also to file header info (e.g. software version, settings...)

        if filehandle != None:

            t_first = filehandle[0][0] # in microseconds
            spike_channel_id = filehandle[0][1]
            cell_count = filehandle[0][2] #number of cells identified
            spike_parameters = filehandle[0][3]
            channel_id = filehandle[0][3][1]
        else:
            t_first = None
            spike_channel_id = None
            cell_count = 0
            spike_parameters =  None
            channel_id = None # TODO: find out channel by .txt header

            self._diagnostic_print('Empty file: No information contained in %s'%filename)

        self.parameters_nse[channel_id] = { 'filename':filename,
                                            't_first': t_first,
                                            'spike_channel_id': spike_channel_id,
                                            'cell_count': cell_count,
                                            'spike_parameters': spike_parameters}


    def __read_nev_header(self, filehandle, filename):
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

        # extract all occurring event marker ids
        event_type_collection = {}
        for event in filehandle:
            event_type_collection[event[4]] = event[10]
        self.parameters_nev['digital_markers'] = copy.deepcopy(event_type_collection)  # entries (marker_id,name)


        # TODO: Extract other important Recording events. But which ones?



    #________________ File Checks __________________________________

    def __ncs_check(self,filehandle):
        '''
        Checks consistency of data in ncs file and raises assertion error if a
        check fails. Detected recording gaps are added to parameter_ncs

        Args:
            filehandle (file object):
                Handle to the already opened .ncs file.
        '''
        # checking number of valid samples per data block
        valid_samples = filehandle[0][3]
        assert all([filehandle[i][3] == valid_samples for i in range(len(filehandle))])

        #time stamps of data pakets
        delta_t = filehandle[1][0] - filehandle[0][0]


        # checking sampling rate of data pakets
        sr0 = filehandle[0][2]
        assert all([filehandle[i][2] == sr0 for i in range(len(filehandle))])

        # checking channel id of data pakets
        channel_id = filehandle[0][1]
        assert all([filehandle[i][1] == channel_id for i in range(len(filehandle))])

        # checking sample count of data pakets
        sample_count = filehandle[0][3]
        assert all([filehandle[i][3] == sample_count for i in range(len(filehandle))])

        #sample count is checked indirectly by the memory occupied by data
#        # checking number of samples in data paket
#        sample_count = len(filehandle[0][4])
#        assert all([len(filehandle[i][4]) == sample_count for i in range(len(filehandle))])

        # unit test
        # time difference between pakets corresponds to number of recorded samples
        # 10**6 due to unit conversion microsec -> sec
        assert delta_t == (sample_count
                           / (self.ncs_time_unit.rescale(pq.s).magnitude * sr0))

        self._diagnostic_print('NCS file check successful.')



    def __nse_check(self,filehandle):
        '''
        Checks consistency of data in ncs file and raises assertion error if a
        check fails.

        Args:
            filehandle (file object):
                Handle to the already opened .nse file.
        '''

        #TODO: Implement this when non-empty .nse files available

        pass

#        self._diagnostic_print('NSE file check successful.')


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



    def __ncs_gap_check(self,filehandle):
        '''
        Checks data blocks of ncs files for consistent starting times
        '''
        #time stamps of data pakets
        delta_t = filehandle[1][0] - filehandle[0][0]
        data_paket_offsets = [filehandle[i+1][0] - filehandle[i][0] for i in range(len(filehandle)-1)]
        if not all(data_paket_offsets == delta_t):
            channel_id = filehandle[0][1]
            if channel_id not in self.parameters_ncs:
                self.parameters_ncs[channel_id] = {}
            if 'gaps' not in self.parameters_ncs[channel_id]:
                self.parameters_ncs[channel_id]['gaps'] = []
            # gap identification by (sample of gap start, duration)
            # gap pakets
            gap_paket_ids = np.where(data_paket_offsets != delta_t)[0]
            for gap_paket_id in gap_paket_ids:

                #skip if this paket starting time is known to be corrupted
                # hoping no corruption and gap occurrs simultaneously #TODO: Check this
                # corrupted time stamp affects two delta_t comparisons:
                if gap_paket_id in self.parameters_ncs[channel_id]['invalid_first_samples'] \
                    or gap_paket_id + 1 in self.parameters_ncs[channel_id]['invalid_first_samples']:
                    continue

                gap_start = filehandle[gap_paket_id][0] + \
                    filehandle[gap_paket_id][3] * 10**6 / filehandle[gap_paket_id][2]
                # gap_stop = gap_id, gap_start, number of valid samples in data packet
                gap_stop = filehandle[gap_paket_id+1][0]
                self.parameters_ncs[channel_id]['gaps'].append((gap_paket_id,gap_start,gap_stop))
                self._diagnostic_print('Detected gap in NCS file at between'
                                        'sample time %i and %i  (last correct '
                                        'paket id %i)'  %(gap_start,gap_stop,
                                                            gap_paket_id))


    def __ncs_invalid_first_sample_check(self,filehandle):
        '''
        Checks data blocks of ncs files for corrupted starting times indicating
        a missing first sample in the data paket. These are then excluded from
        the gap check, but ignored for further analysis.
        '''
        channel_id = filehandle[0][1]
        self.parameters_ncs[channel_id]['invalid_first_samples'] = []

        # collating starting times of data pakets
        t_starts = np.array([f[0] for f in filehandle])
        #checking if first bit of timestamp is 1, which indicates error
        invalid_paket_ids = np.where(t_starts >= 3.6028797018963968*10**16)[0]
        if len(invalid_paket_ids)>0:
            warnings.warn('Invalid first sample(s) detected in ncs file'
                            '(paket id(s) %i)! This error is ignored in'
                            'subsequent routines.'%(invalid_paket_ids))
            self.parameters_ncs[channel_id]['invalid_first_samples'] = invalid_paket_ids

            #checking consistency of data around corrupted paket time
            for invalid_paket_id in invalid_paket_ids:
                if invalid_paket_id < 2 or invalid_paket_id > len(filehandle) -2:
                    raise ValueError('Corrupted ncs data paket at the beginning'
                                        'or end of file.')
                elif (t_starts[invalid_paket_id+1] - t_starts[invalid_paket_id-1]
                != 2* (t_starts[invalid_paket_id-1] - t_starts[invalid_paket_id-2])):
                    raise ValueError('Starting times of ncs data pakets around'
                                     'corrupted data paket are not consistent!')




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