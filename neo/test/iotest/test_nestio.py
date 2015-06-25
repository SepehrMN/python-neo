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

from neo.io.nestio import NestIO
from neo.test.iotest.common_io_test import BaseTestIO
import quantities as pq
import numpy as np


class TestNestIO(BaseTestIO, unittest.TestCase):
    ioclass = NestIO
    files_to_test = []
    files_to_download = []

    def test_read_analogsignalarray(self):
        '''
        Tests reading files in the 4 different formats:
        - with GIDs, with times as floats
        - with GIDs, with times as integers in time steps
        '''

        r = NestIO(filename='nest_test_files/withgidT-time_in_stepsF-1259-0.dat')
        r.read_analogsignalarray(gid=1, t_stop=1000.*pq.ms,
                                 sampling_period=pq.ms, lazy=False,
                                 id_column=0, time_column=1,
                                 value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                       sampling_period=pq.ms, lazy=False, id_column=0,
                       time_column=1, value_column=2,
                       value_type='V_m')

        r = NestIO(filename='nest_test_files/withgidT-time_in_stepsT-1261-0.dat')
        r.read_analogsignalarray(gid=1, t_stop=1000.*pq.ms,
                                 time_unit=pq.CompoundUnit('0.1*ms'),
                                 sampling_period=pq.ms, lazy=False,
                                 id_column=0, time_column=1,
                                 value_column=2, value_type='V_m')
        r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                       time_unit=pq.CompoundUnit('0.1*ms'),
                       sampling_period=pq.ms, lazy=False, id_column=0,
                       time_column=1, value_column=2,
                       value_type='V_m')

    def test_id_column_none_multiple_neurons(self):
        '''
        Tests if function correctly raises an error if the user tries to read
        from a file which does not contain neuron IDs, but data for multiple
        neurons.
        '''

        r = NestIO(filename='nest_test_files/withgidF-time_in_stepsF-1258-0.dat')
        with self.assertRaises(ValueError):
            r.read_analogsignalarray(t_stop=1000.*pq.ms, lazy=False,
                                     sampling_period=pq.ms,
                                     id_column=None, time_column=0,
                                     value_column=1)
            r.read_segment(t_stop=1000.*pq.ms, lazy=False,
                           sampling_period=pq.ms, id_column=None,
                           time_column=0, value_column=1)

        r = NestIO(filename='nest_test_files/withgidF-time_in_stepsT-1260-0.dat')
        with self.assertRaises(ValueError):
            r.read_analogsignalarray(t_stop=1000.*pq.ms,
                                     time_unit=pq.CompoundUnit('0.1*ms'),
                                     lazy=False, id_column=None,
                                     time_column=0, value_column=1)
            r.read_segment(t_stop=1000.*pq.ms,
                           time_unit=pq.CompoundUnit('0.1*ms'), lazy=False,
                           id_column=None, time_column=0, value_column=1)



    def test_values(self):
        '''
        Tests if the function returns the correct values.
        '''

        id_to_test = 1
        r = NestIO(filename='nest_test_files/withgidT-time_in_stepsF-1259-0.dat')
        seg = r.read_segment(gid_list=[id_to_test],
                             t_stop=1000.*pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column=0, time_column=1,
                             value_column=2, value_type='V_m')

        dat = np.loadtxt('nest_test_files/withgidT-time_in_stepsF-1259-0.dat')
        target_data = dat[:, 2][np.where(dat[:, 0] == id_to_test)]
        st = seg.analogsignalarrays[0]
        np.testing.assert_array_equal(st.magnitude, target_data)

    def test_read_segment(self):
        '''
        Tests if signals are correctly stored in a segment.
        '''

        r = NestIO(filename='nest_test_files/withgidT-time_in_stepsF-1259-0.dat')

        id_list_to_test = range(1, 10)
        seg = r.read_segment(gid_list=id_list_to_test,
                             t_stop=1000.*pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column=0, time_column=1,
                             value_column=2, value_type='V_m')

        self.assertTrue(len(seg.analogsignalarrays) == len(id_list_to_test))

        id_list_to_test = []
        seg = r.read_segment(gid_list=id_list_to_test,
                             t_stop=1000.*pq.ms,
                             sampling_period=pq.ms, lazy=False,
                             id_column=0, time_column=1,
                             value_column=2, value_type='V_m')

        self.assertTrue(len(seg.analogsignalarrays) == 50)

    def test_wrong_input(self):
        '''
        Tests two cases of wrong user input, namely
        - User does not specify a value column
        - User does not make any specifications
        - User does not define sampling_period as a unit
        - User specifies a non-default value type without
          specifying a value_unit
        - User specifies t_start < 1.*sampling_period
        '''

        r = NestIO(filename='nest_test_files/withgidT-time_in_stepsF-1259-0.dat')
        with self.assertRaises(ValueError):
            r.read_segment(t_stop=1000.*pq.ms, lazy=False,
                           id_column=0, time_column=1)
        with self.assertRaises(ValueError):
            r.read_segment()
        with self.assertRaises(ValueError):
            r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                           sampling_period=1.*pq.ms, lazy=False,
                           id_column=0, time_column=1,
                           value_column=2, value_type='V_m')

        with self.assertRaises(ValueError): 
            r.read_segment(gid_list=[1], t_stop=1000.*pq.ms,
                           sampling_period=pq.ms, lazy=False,
                           id_column=0, time_column=1,
                           value_column=2, value_type='U_mem')

        with self.assertRaises(ValueError): 
            r.read_segment(gid_list=[1], t_start=0.*pq.ms, t_stop=1000.*pq.ms,
                           sampling_period=pq.ms, lazy=False,
                           id_column=0, time_column=1,
                           value_column=2, value_type='V_m')


    def test_t_start_t_stop(self):
        r = NestIO(filename='nest_test_files/withgidT-time_in_stepsF-1259-0.dat')

        t_stop_targ = 100.*pq.ms
        t_start_targ = 50.*pq.ms

        seg = r.read_segment(gid_list=[], t_start=t_start_targ,
                             t_stop=t_stop_targ, lazy=False,
                             id_column=0, time_column=1,
                             value_column=2, value_type='V_m')
        sts = seg.analogsignalarrays
        for st in sts:
            self.assertTrue(st.t_start == t_start_targ)
            self.assertTrue(st.t_stop == t_stop_targ)

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestNestIO('test_read_analogsignalarray'))
    # unittest.TextTestRunner(verbosity=2).run(suite)
