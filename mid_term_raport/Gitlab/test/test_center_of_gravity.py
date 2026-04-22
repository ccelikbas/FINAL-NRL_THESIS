import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from center_of_gravity import xcg_start, xcg_shift

class TestCenterofGravity(unittest.TestCase):

    # def Setup(self):

    def test_xcg_start(self):
        xcg_start_meas = xcg_start()[0]
        self.assertAlmostEqual(xcg_start_meas, 7.147653776, delta=0.01)

    def test_xcg_shift(self):
        xcg_shift_result = xcg_shift()
        xcg_shift_expected = -0.04734
        self.assertAlmostEqual(xcg_shift_expected, xcg_shift_result, delta = 0.005)  # add assertion here


if __name__ == '__main__':
    unittest.main()
