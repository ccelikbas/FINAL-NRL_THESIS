import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from data_processing import get_cl_cd, calc_C_m_delta, plot_trim_curve
from data_processing import *
class TestDataProcessing(unittest.TestCase):


    def test_C_m_delta(self):
        C_m_delta = calc_C_m_delta(7750 * ft_to_m, 160 * kts_to_ms, cel_to_k - 2, -0.6 * pi/180)
        
        C_m_delta_manual = -1.0698
        self.assertAlmostEqual(C_m_delta, C_m_delta_manual, delta=0.001)

    def test_C_M_alpha(self):
        C_M_alpha = plot_trim_curve()

        C_m_alpha_manual = -0.4964
        self.assertAlmostEqual(C_M_alpha, C_m_alpha_manual, delta=0.001)

    def test_C_L(self):
        C_L_alpha_meas = get_cl_cd() * 57.2958
        C_L_alpha_manual = 4.21
        self.assertAlmostEqual(C_L_alpha_meas, C_L_alpha_manual, delta=0.01)    

if __name__ == '__main__':
    unittest.main()
