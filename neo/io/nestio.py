# -*- coding: utf-8 -*-
"""
Class for reading output files of NEST.

Depends on: numpy, quantities

Supported: Read

Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import numpy as np
import quantities as pq

from neo.io import GdfIO
from neo.core import Segment, SpikeTrain, AnalogSignalArray

value_types = {'V': pq.mV,
               'I': pq.pA,
               'g': pq.CompoundUnit("10^-9*S")}


class NestIO(GdfIO):

    """
    Class for reading GDF files, e.g., the spike output of NEST.
    TODO 
    Usage:
        TODO
    """
    
    is_readable = True # This class can only read data
    is_writable = False

    supported_objects = [SpikeTrain, AnalogSignalArray]
    readable_objects = [SpikeTrain, AnalogSignalArray]

    has_header = False
    is_streameable = False

    # do not supported write so no GUI stuff
    write_params = None

    name = 'nest'
    extensions = ['gdf', 'dat']
    mode = 'file'


    def __init__(self, filename=None):
        """
        Parameters
        ----------
            filename: string, default=None
                The filename.
        """
        GdfIO.__init__(self, filename)
        self.filename = filename


    def __read_analogsinalarrays(self, gid_list, time_unit, t_start,
                                 t_stop, sampling_period=None,
                                 id_column=0, time_column=1,
                                 value_column=2, value_type=None,
                                 value_unit=None):
        """
        Internal function called by read_analogsignalarray() and read_segment().
        """

        # load GDF data
        f = open(self.filename)
        # read the first line to check the data type (int or float) of the spike
        # times, assuming that only the column of time stamps may contain floats
        line = f.readline()
        if '.' not in line:
            data = np.loadtxt(self.filename, dtype=np.int32)
        else:
            data = np.loadtxt(self.filename, dtype=np.float)


        # check loaded data and given arguments
        if len(data.shape) < 2 and id_column is not None:
            raise ValueError('File does not contain neuron IDs but '
                             'id_column specified to '+str(id_column)+'.')

        if time_column is None:
            raise ValueError('No time column provided.')

        if value_column is None:
            raise ValueError('No value column provided.')

        if value_type is not None and value_type.split('_')[0] in value_types:
            value_unit = value_types[value_type.split('_')[0]]
        elif not isinstance(value_unit, pq.UnitQuantity):
            raise ValueError('No value unit or standard value type specified.')

        if None in gid_list and id_column is not None:
            raise ValueError('No neuron IDs specified but file contains '
                             'neuron IDs in column '+str(id_column)+'.'
                             ' Specify empty list to ' 'retrieve'
                             ' spiketrains of all neurons.')

        if gid_list != [None] and id_column is None:
            raise ValueError('Specified neuron IDs to '
                             'be '+str(gid_list)+','
                             ' but no ID column specified.')

        if t_stop is None:
            raise ValueError('No t_stop specified.')

        if not isinstance(t_stop,pq.quantity.Quantity):
            raise TypeError('t_stop (%s) is not a quantity.'%(t_stop))

        if not isinstance(t_start,pq.quantity.Quantity):
            raise TypeError('t_start (%s) is not a quantity.'%(t_start))

        # assert that no single column is assigned twice
        column_list = [id_column, time_column, value_column]
        if len(np.unique(column_list)) < 3:
            raise ValueError('1 or more columns have been specified to '
            'contain the same data. Columns were specified to '
            '%s.'%(column_list))

        # get neuron gid_list
        if gid_list == []:
            gid_list = np.unique(data[:, id_column]).astype(int)

        # # get consistent dimensions of data
        # if len(data.shape)<2:
        #     data = data.reshape((-1,1))

        # use only data from the time interval between t_start and t_stop
        data = data[np.where(np.logical_and( data[:, time_column] >=
                                             t_start.rescale(time_unit).magnitude,
                                             data[:, time_column] <
                                             t_stop.rescale(time_unit).magnitude))]

        # create an empty list of signals and fill in the signals for each
        # GID in gid_list
        analogsignal_list = []
        for i in gid_list:
            # find the signal for each neuron ID
            if id_column is not None:
                signal = data[np.where(data[:, id_column] == i)][:, value_column]
                times = data[np.where(data[:, id_column] == i)][:, time_column]
            else:
                signal = data[:, value_column]
                times = data[:, time_column]
            
            if id_column is None and len(np.unique(times)) < len(times):
                raise ValueError('No ID column specified but recorded '
                                 'from multiple neurons.')

            if sampling_period is None:
                # Could be inaccurate because of floats
                dt = times[1]-times[0]
                sampling_period = pq.CompoundUnit(str(dt)+'*'+time_unit.units.u_symbol)  
            elif not isinstance(sampling_period, pq.UnitQuantity):
                raise ValueError("sampling_period is not specified as a unit.")

            # check if signal has the correct length
            assert(len(signal) ==
                   t_stop.rescale(sampling_period).magnitude -
                   t_start.rescale(sampling_period).magnitude-1)

            # create AnalogSinalArray objects and annotate them with the neuron ID
            analogsignal_list.append(AnalogSignalArray(signal*value_unit,
                                                       sampling_period=sampling_period,
                                                       t_start=t_start,
                                                       annotations={'id':
                                                                    i, 'type' :
                                                                    value_type}))

        return analogsignal_list
        
    def read_segment(self, lazy=False, cascade=True, gid_list=None,
                     time_unit=pq.ms, t_start=0.*pq.ms, t_stop=None,
                     sampling_period=None, id_column=0, time_column=1,
                     value_column=2, value_type=None, value_unit=None):
        """
        Read a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gid_list : list, default: None
            A list of GDF IDs of which to return SpikeTrain(s). gid_list must
            be specified if the GDF file contains neuron IDs, the default None
            then raises an error. Specify an empty list [] to retrieve the spike
            trains of all neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.

        Returns
        -------
        seg : Segment
            The Segment contains one SpikeTrain for each ID in gid_list.      
        """

        # __read_spiketrains() needs a list of IDs
        if gid_list is None:
            gid_list = [None]

        # create an empty Segment and fill in the spike trains
        seg = Segment()
        seg.analogsignalarrays = self.__read_analogsinalarrays(gid_list,
                                                        time_unit, t_start,
                                                        t_stop,
                                                        sampling_period=sampling_period,
                                                        id_column=id_column,
                                                        time_column=time_column,
                                                        value_type=value_type,
                                                        value_unit=value_unit)

        return seg

    def read_analogsignalarray( self, lazy=False, cascade=True,
                                gid=None, time_unit=pq.ms, t_start=0 * pq.ms,
                                t_stop=None, sampling_period=None, id_column=0,
                                time_column=1, value_column=2, value_type=None,
                                value_unit=None):
        """
        Read SpikeTrain with specified neuron ID from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gdf_id : int, default: None
            The GDF ID of the returned SpikeTrain. gdf_id must be specified if
            the GDF file contains neuron IDs, the default None then raises an
            error. Specify an empty list [] to retrieve the spike trains of all
            neurons.
        time_unit : Quantity (time), optional, default: quantities.ms
            The time unit of recorded time stamps.
        t_start : Quantity (time), optional, default: 0 * pq.ms
            Start time of SpikeTrain.
        t_stop : Quantity (time), default: None
            Stop time of SpikeTrain. t_stop must be specified, the default None
            raises an error.
        id_column : int, optional, default: 0
            Column index of neuron IDs.
        time_column : int, optional, default: 1
            Column index of time stamps.

        Returns
        -------
        spiketrain : SpikeTrain
            The requested SpikeTrain object with an annotation 'id'
            corresponding to the gdf_id parameter.
        """

        # __read_spiketrains() needs a list of IDs
        return self.__read_analogsinalarrays([gid], time_unit,
                                             t_start, t_stop,
                                             sampling_period=sampling_period,
                                             id_column=id_column,
                                             time_column=time_column,
                                             value_column=value_column,
                                             value_type=value_type,
                                             value_unit=value_unit)[0]
