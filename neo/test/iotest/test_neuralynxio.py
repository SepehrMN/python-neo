# -*- coding: utf-8 -*-
"""
Tests of neo.io.blackrockio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os
import struct
import sys
import tempfile

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
import quantities as pq

from neo import NeuralynxIO, AnalogSignalArray, Segment, SpikeTrain, EventArray, Unit
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io import tools
from neo.core import Segment


#~ class testRead(unittest.TestCase):
    #~ """Tests that data can be read from KlustaKwik files"""
    #~ def test1(self):
        #~ """Tests that data and metadata are read correctly"""
        #~ pass
    #~ def test2(self):
        #~ """Checks that cluster id autosets to 0 without clu file"""
        #~ pass
        #~ dirname = os.path.normpath('./files_for_tests/klustakwik/test2')
        #~ kio = neo.io.KlustaKwikIO(filename=os.path.join(dirname, 'base2'),
            #~ sampling_rate=1000.)
        #~ seg = kio.read()
        #~ seg = block.segments[0]
        #~ self.assertEqual(len(seg.spiketrains), 1)
        #~ self.assertEqual(seg.spiketrains[0].name, 'unit 0 from group 5')
        #~ self.assertEqual(seg.spiketrains[0].annotations['cluster'], 0)
        #~ self.assertEqual(seg.spiketrains[0].annotations['group'], 5)
        #~ self.assertEqual(seg.spiketrains[0].t_start, 0.0)
        #~ self.assertTrue(np.all(seg.spiketrains[0].times == np.array(
            #~ [0.026, 0.122, 0.228])))


class CommonTests(BaseTestIO, unittest.TestCase):
    ioclass = NeuralynxIO
    read_and_write_is_bijective = False
    hash_conserved_when_write_read = False

    files_to_test = [
        # 'testsession/STet3a.nse',
        # 'testsession/STet3b.nse',
        # 'testsession/Tet3a.ncs',
        # 'testsession/Tet3b.ncs',
        # 'plaindata/STet3a.txt',
        # 'plaindata/STet3b.txt',
        # 'plaindata/Tet3a.txt',
        # 'plaindata/Tet3b.txt'
        ]

    files_to_download = [
        # 'testsession/STet3a.nse',
        # 'testsession/STet3b.nse',
        # 'testsession/Tet3a.ncs',
        # 'testsession/Tet3b.ncs',
        # 'plaindata/STet3a.txt',
        # 'plaindata/STet3b.txt',
        # 'plaindata/Tet3a.txt',
        # 'plaindata/Tet3b.txt'
        ]

    local_test_dir = None


@unittest.skipIf(sys.version_info[0] > 2, "not Python 3 compatible")
class testRead(unittest.TestCase):
    def setUp(self):
        self.sn = ('/home/julia/repositories/python/python-neo/neo/test/'
                   'iotest/neuralynx_test_files/testsession')
        self.pd = ('/home/julia/repositories/python/python-neo/neo/test/'
                   'iotest/neuralynx_test_files/plaindata')
#        self.sn = os.path.join(tempfile.gettempdir(),
#                               'files_for_testing_neo',
#                               'blackrock/test2/test.ns5')
#         self.pd = os.path.join(tempfile.gettempdir(),
# #                               'files_for_testing_neo',
# #                               'blackrock/test2/test.ns5')
        if not os.path.exists(self.sn):
            raise unittest.SkipTest('data file does not exist:' + self.sn)

    def test_read_block(self):
        """Read data in a certain time range into one block"""
        t_start,t_stop = 3*pq.s,4*pq.s

        nio = NeuralynxIO(self.sn)
        block = nio.read_block(t_starts=[t_start], t_stops=[t_stop])
        self.assertEqual(len(nio.parameters_ncs), 2)
        self.assertTrue( {'event_id': 11, 'name': 'Starting Recording', 'nttl': 0} in nio.parameters_nev['Events.nev']['event_types'])

        # Everything put in one segment
        self.assertEqual(len(block.segments), 1)
        seg = block.segments[0]
        self.assertEqual(len(seg.analogsignalarrays), 2)

        self.assertTrue(all([(seg.analogsignalarrays[i].sampling_rate.units ==
                                pq.CompoundUnit('32*kHz')) for i in range(2)]))
        self.assertTrue(all([seg.analogsignalarrays[i].t_start == t_start
                                for i in range(2)]))
        self.assertTrue(all([seg.analogsignalarrays[i].t_stop == t_stop
                                for i in range(2)]))
        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        block = nio.read_block(lazy=True)
        self.assertTrue(len(block.segments[0].analogsignalarrays[0])==0)
        self.assertTrue(len(block.segments[0].spiketrains[0])==0)

        block = nio.read_block(cascade=False)
        self.assertTrue(len(block.segments)==0)

        block = nio.read_block(electrode_list=[0])
        self.assertTrue(len(block.segments[0].analogsignalarrays)==1)

        block = nio.read_block(t_starts=None,t_stops=None,events=True,waveforms=True)
        self.assertTrue(len(block.segments[0].analogsignalarrays)==2)
        self.assertTrue(len(block.segments[0].spiketrains)==2)
        self.assertTrue(len(block.segments[0].spiketrains[0].waveforms)>0)
        self.assertTrue(len(block.segments[0].eventarrays)>0)
        self.assertTrue(len(block.recordingchannelgroups[1].units)>0)


    def test_read_segment(self):
        """Read data in a certain time range into one block"""
        # t_start,t_stop = 2*pq.s,10*pq.s

        nio = NeuralynxIO(self.sn)
        seg = nio.read_segment(t_start=None, t_stop=None)

        self.assertEqual(len(seg.analogsignalarrays), 2)

        self.assertTrue(all([(seg.analogsignalarrays[i].sampling_rate.units ==
                                pq.CompoundUnit('32*kHz')) for i in range(2)]))

        self.assertEqual(len(seg.spiketrains), 2)

        # Testing different parameter combinations
        seg = nio.read_segment(lazy=True)
        self.assertTrue(len(seg.analogsignalarrays[0])==0)
        self.assertTrue(len(seg.spiketrains[0])==0)

        seg = nio.read_segment(cascade=False)
        self.assertTrue(len(seg.analogsignalarrays)==0)
        self.assertTrue(len(seg.spiketrains)==0)

        seg = nio.read_segment(electrode_list=[0])
        self.assertTrue(len(seg.analogsignalarrays)==1)

        seg = nio.read_segment(t_start=None,t_stop=None,events=True,waveforms=True)
        self.assertTrue(len(seg.analogsignalarrays)==2)
        self.assertTrue(len(seg.spiketrains)==2)
        self.assertTrue(len(seg.spiketrains[0].waveforms)>0)
        self.assertTrue(len(seg.eventarrays)>0)




    def test_read_ncs_data(self):
        t_start,t_stop = 0,500*512 # in samples

        nio = NeuralynxIO(self.sn)
        seg = Segment('testsegment')

        for el_id,filename in [(0,'Tet3a'),(1,'Tet3b')]:
            nio.read_ncs(filename,seg,t_start=t_start, t_stop=t_stop)
            anasig = seg.filter({'electrode_id':el_id},objects=AnalogSignalArray)[0]

            target_data = np.zeros((500,512))
            with open(self.pd + '/%s.txt'%filename) as file:
                for i, line in enumerate(file):
                    line = line.strip('\xef\xbb\xbf')
                    entries = line.split()
                    target_data[i,:] = entries[5:]

            target_data = target_data.reshape((-1)) * 3.05185e-08

            self.assertTrue(all(target_data==anasig.magnitude))


    def test_read_nse_data(self):
        t_start,t_stop = 0,500*512 # in samples

        nio = NeuralynxIO(self.sn)
        seg = Segment('testsegment')

        for el_id,filename in [(0,'STet3a'),(1,'STet3b')]:
            nio.read_nse(filename,seg,t_start=t_start, t_stop=t_stop,waveforms=True)
            spiketrain = seg.filter({'electrode_id':el_id},objects=SpikeTrain)[0]

            target_data = np.zeros((500,32))
            timestamps = np.zeros((500))
            with open(self.pd + '/%s.txt'%filename) as file:
                for i, line in enumerate(file):
                    line = line.strip('\xef\xbb\xbf')
                    entries = line.split()
                    target_data[i,:] = entries[12:]
                    timestamps[i] = entries[1]

            timestamps = np.array(timestamps)*pq.microsecond - nio.parameters_global['t_start']
             # masking only requested spikes
            mask=np.where(timestamps<t_stop/32000.*pq.s)
            timestamps = timestamps[mask]
            target_data = target_data[mask]

            np.testing.assert_array_equal(timestamps,spiketrain.magnitude)
            np.testing.assert_array_equal(target_data,spiketrain.waveforms)


    def test_read_nev_data(self):
        t_start,t_stop = 0*pq.s,1000*pq.s

        nio = NeuralynxIO(self.sn)
        seg = Segment('testsegment')

        filename = 'Events'
        nio.read_nev(filename + '.nev',seg,t_start=t_start, t_stop=t_stop)

        timestamps = []
        nttls = []
        names = []
        event_ids = []

        with open(self.pd + '/%s.txt'%filename) as file:
            for i, line in enumerate(file):
                line = line.strip('\xef\xbb\xbf')
                entries = line.split('\t')
                nttls.append(int(entries[3],2))
                timestamps.append(int(entries[1]))
                names.append(entries[4].rstrip('\r\n'))
                event_ids.append(int(entries[2]))

        timestamps = np.array(timestamps) * pq.microsecond -  nio.parameters_global['t_start']
         # masking only requested spikes
        mask=np.where(timestamps<t_stop)[0]

        # return if no event fits criteria
        if len(mask)==0:
            return
        timestamps = timestamps[mask]
        nttls = np.asarray(nttls)[mask]
        names = np.asarray(names)[mask]
        event_ids = np.asarray(event_ids)[mask]

        for i in range(len(timestamps)):
            events = seg.filter({'nttl':nttls[i]},objects=EventArray)
            events = [e for e in events if (e.annotations['marker_id']==event_ids[i] and e.labels==names[i])]
            self.assertTrue(len(events)==1)
            self.assertTrue(timestamps[i] in events[0].times)


    def test_read_ntt_data(self):
        pass
    #TODO: Implement test_read_ntt_data one ntt files are available


    def test_gap_handling(self):
        nio = NeuralynxIO(self.sn)

        block = nio.read_block(t_starts=None,t_stops=None)

        # known gap values
        n_gaps = 3

        self.assertTrue(len(block.segments)==n_gaps)
        self.assertTrue(len(block.recordingchannelgroups[0].recordingchannels)==2) # == number of electrode
        self.assertTrue(len(block.recordingchannelgroups[1].units)==n_gaps)
        self.assertTrue(len(block.recordingchannelgroups[2].units)==n_gaps)

        self.assertTrue(len(block.recordingchannelgroups[1].recordingchannels[0].analogsignals)==n_gaps)
        self.assertTrue(len(block.recordingchannelgroups[1].recordingchannels[0].analogsignals)==n_gaps)


if __name__ == '__main__':
    unittest.main()
