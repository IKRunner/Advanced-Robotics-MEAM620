# Imports
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

from code.complementary_filter import complementary_filter_update


class TestBase(unittest.TestCase):

    def complementary_filter_update_test(self, fname):
        with open(fname, 'r') as file:

            print ('running : ' + fname)

            d = json.load(file)

            # Run a test
            R0 = Rotation.from_quat(d['initial_rotation'])
            w = np.array(d['angular_velocity'])
            a = np.array(d['linear_acceleration'])
            dt = d['dt']

            output_rotation = Rotation.from_quat(d['output_rotation'])

            rout = complementary_filter_update(R0, w, a, dt)

            temp = rout.inv() * output_rotation

            self.assertTrue(temp.magnitude() < 1e-4, 'failed ' + fname)

    # test complementary_filter_update
    def test_complementary_filter_update_00(self):
        self.complementary_filter_update_test('test_complementary_filter_00.json')

    def test_complementary_filter_update_01(self):
        self.complementary_filter_update_test('test_complementary_filter_01.json')

    def test_complementary_filter_update_02(self):
        self.complementary_filter_update_test('test_complementary_filter_02.json')

    def test_complementary_filter_update_03(self):
        self.complementary_filter_update_test('test_complementary_filter_03.json')

    def test_complementary_filter_update_04(self):
        self.complementary_filter_update_test('test_complementary_filter_04.json')

if __name__ == '__main__':
    unittest.main()
