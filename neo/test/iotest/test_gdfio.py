# -*- coding: utf-8 -*-
"""
Tests of neo.io.exampleio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.io.gdfio import GdfIO
from neo.test.iotest.common_io_test import BaseTestIO
import quantities as pq
import numpy as np


class TestGdfIO(BaseTestIO, unittest.TestCase):
    ioclass = GdfIO
    files_to_test = []
    files_to_download = []

    def test_read_spiketrain(self):
        '''
        Tests reading files in the 4 different formats:
        - without GIDs, with times as floats
        - without GIDs, with times as integers in time steps
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        '''

        r = GdfIO(filename='gdf_test_files/withgidF-time_in_stepsF-1254-0.gdf')
        r.read_spiketrain(t_stop=1000.*pq.ms, lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_stop=1000.*pq.ms, lazy=False, id_column=None,
                       time_column=0)

        r = GdfIO(filename='gdf_test_files/withgidF-time_in_stepsT-1256-0.gdf')
        r.read_spiketrain(t_stop=1000.*pq.ms,
                          time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                          id_column=None, time_column=0)
        r.read_segment(t_stop=1000.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                       id_column=None, time_column=0)

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        r.read_spiketrain(gdf_id=1, t_stop=1000.*pq.ms, lazy=False,
                          id_column=0, time_column=1)
        r.read_segment(gdf_id_list=[1], t_stop=1000.*pq.ms,
                       lazy=False, id_column=0, time_column=1)

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsT-1257-0.gdf')
        r.read_spiketrain(gdf_id=1, t_stop=1000.*pq.ms, lazy=False,
                          id_column=0, time_column=1)
        r.read_segment(gdf_id_list=[1], t_stop=1000.*pq.ms,
                       lazy=False, id_column=0, time_column=1)

    def test_read_integer(self):
        '''
        Tests if spike times are actually stored as integers if they
        are stored in time steps in the file.
        '''

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsT-1257-0.gdf')
        st = r.read_spiketrain(gdf_id=1, t_stop=1000.*pq.ms,
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.int32)
        seg = r.read_segment(gdf_id_list=[1], t_stop=1000.*pq.ms,
                             lazy=False, id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(all([st.magnitude.dtype == np.int32 for st in sts]))

    def test_read_float(self):
        '''
        Tests if spike times are stored as floats if they
        are stored as floats in the file.
        '''

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        st = r.read_spiketrain(gdf_id=1, t_stop=1000.*pq.ms,
                               lazy=False, id_column=0, time_column=1)
        self.assertTrue(st.magnitude.dtype == np.float)
        seg = r.read_segment(gdf_id_list=[1], t_stop=1000.*pq.ms,
                             lazy=False, id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(all([s.magnitude.dtype == np.float for s in sts]))

    def test_values(self):
        '''
        Tests if the routine loads the correct numbers from the file.
        '''

        id_to_test = 1
        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        seg = r.read_segment(gdf_id_list=[id_to_test],
                             t_stop=1000.*pq.ms, lazy=False,
                             id_column=0, time_column=1)

        dat = np.loadtxt('gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        target_data = dat[:, 1][np.where(dat[:, 0]==id_to_test)]

        st = seg.spiketrains[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        '''
        Tests if spiketrains are correctly stored in a segment.
        '''

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')

        id_list_to_test = range(1,10)
        seg = r.read_segment(gdf_id_list=id_list_to_test,
                             t_stop=1000.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gdf_id_list=id_list_to_test,
                             t_stop=1000.*pq.ms, lazy=False,
                             id_column=0, time_column=1)
        self.assertTrue(len(seg.spiketrains) == 50)

    def test_wrong_input(self):
        '''
        Tests two cases of wrong user input, namely
        - User does not specify neuron IDs although the file contains IDs
        - User does not make any specifications
        '''

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')
        with self.assertRaises(ValueError):
            r.read_segment(t_stop=1000.*pq.ms, lazy=False,
                           id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment()

    def test_t_start_t_stop(self):
        '''
        Tests if the t_start and t_stop arguments are correctly processed.
        '''

        r = GdfIO(filename='gdf_test_files/withgidT-time_in_stepsF-1255-0.gdf')

        t_stop_targ = 100.*pq.ms
        t_start_targ = 50.*pq.ms

        seg = r.read_segment(gdf_id_list=[], t_start=t_start_targ,
                             t_stop=t_stop_targ, lazy=False,
                             id_column=0, time_column=1)
        sts = seg.spiketrains
        self.assertTrue(np.max([np.max(st.magnitude) for st in sts]) <
                        t_stop_targ.rescale(sts[0].times.units).magnitude)
        self.assertTrue(np.min([np.min(st.magnitude) for st in sts])
                        >= t_start_targ.rescale(sts[0].times.units).magnitude)

if __name__ == "__main__":
    unittest.main()
