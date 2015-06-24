# -*- coding: utf-8 -*-
"""
Class for reading GDF data files, e.g., the spike output of NEST.

Depends on: numpy, quantities

Supported: Read

Authors: Julia Sprenger, Maximilian Schmidt, Johanna Senk

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, SpikeTrain


class GdfIO(BaseIO):

    """
    Class for reading GDF files, e.g., the spike output of NEST.

    Usage:
        TODO
    """
    
    is_readable = True # This class can only read data
    is_writable = False

    supported_objects = [SpikeTrain]
    readable_objects = [SpikeTrain]

    has_header = False
    is_streameable = False

    # do not supported write so no GUI stuff
    write_params = None

    name = 'gdf'
    extensions = ['gdf']
    mode = 'file'


    def __init__(self, filename=None):
        """
        Parameters
        ----------
            filename: string, default=None
                The filename.
        """
        BaseIO.__init__(self)
        self.filename = filename


    def __read_spiketrains(self, gdf_id_list, time_unit,
                           t_start, t_stop, id_column=0,
                           time_column=1):
        """
        Internal function called by read_spiketrain() and read_segment().
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

        # assert that there are spike times in the file
        if time_column is None:
            raise ValueError('Time column is None. No spike times to '
                             'be read in.')

        if None in gdf_id_list and id_column is not None:
            raise ValueError('No neuron IDs specified but file contains '
                             'neuron IDs in column '+str(id_column)+'.'
                             ' Specify empty list to ' 'retrieve'
                             ' spiketrains of all neurons.')

        if gdf_id_list != [None] and id_column is None:
            raise ValueError('Specified neuron IDs to '
                             'be '+str(gdf_id_list)+','
                             ' but file does not contain neuron IDs.')

        if t_stop is None:
            raise ValueError('No t_stop specified.')

        if not isinstance(t_stop,pq.quantity.Quantity):
            raise TypeError('t_stop (%s) is not a quantity.'%(t_stop))

        if not isinstance(t_start,pq.quantity.Quantity):
            raise TypeError('t_start (%s) is not a quantity.'%(t_start))

        # assert that no single column is assigned twice
        if id_column == time_column:
            raise ValueError('1 or more columns have been specified to '
                             'contain the same data.')

        # get neuron gdf_id_list
        if gdf_id_list == []:
            gdf_id_list = np.unique(data[:, id_column]).astype(int)

        # get consistent dimensions of data
        if len(data.shape)<2:
            data = data.reshape((-1,1))


        # use only data from the time interval between t_start and t_stop
        data = data[np.where(np.logical_and(
                    data[:,time_column]>=t_start.rescale(time_unit).magnitude,
                    data[:,time_column]<t_stop.rescale(time_unit).magnitude))]

        # create an empty list of spike trains and fill in the trains for each
        # GDF ID in gdf_id_list
        spiketrain_list = []
        for i in gdf_id_list:
            # find the spike times for each neuron ID
            if id_column is not None:
                train = data[np.where(data[:, id_column] == i)][:,time_column]
            else:
                train = data[:,time_column]
            # create SpikeTrain objects and annotate them with the neuron ID
            spiketrain_list.append(SpikeTrain(
                train, units=time_unit, t_start=t_start, t_stop=t_stop,
                annotations={'id': i}))

        return spiketrain_list
        

    def read_segment(self, lazy=False, cascade=True,
                     gdf_id_list=None, time_unit=pq.ms, t_start=0.*pq.ms,
                     t_stop=None, id_column=0, time_column=1):
        """
        Read a Segment which contains SpikeTrain(s) with specified neuron IDs
        from the GDF data.

        Parameters
        ----------
        lazy : bool, optional, default: False
        cascade : bool, optional, default: True
        gdf_id_list : list, default: None
            A list of GDF IDs of which to return SpikeTrain(s). gdf_id_list must
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
            The Segment contains one SpikeTrain for each ID in gdf_id_list.      
        """

        # __read_spiketrains() needs a list of IDs
        if gdf_id_list is None:
            gdf_id_list = [None]

        # create an empty Segment and fill in the spike trains
        seg = Segment()
        seg.spiketrains = self.__read_spiketrains(gdf_id_list,
                                                  time_unit, t_start,
                                                  t_stop,
                                                  id_column=id_column,
                                                  time_column=time_column)

        return seg


    def read_spiketrain(
            self, lazy=False, cascade=True, gdf_id=None,
            time_unit=pq.ms, t_start=0 * pq.ms, t_stop=None,
            id_column=0, time_column=1):
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
        return self.__read_spiketrains([gdf_id], time_unit,
                                       t_start, t_stop,
                                       id_column=id_column,
                                       time_column=time_column)[0]
