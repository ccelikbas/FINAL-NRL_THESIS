import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Cit_par import *
from state_space_matrices import state_space_matrices_asymmetric, state_space_matrices_symmetric

V0 = 80  # m/s
hp0 = 0
th0 = 0.01
m = 6000  # kg

class TestStateSpace(unittest.TestCase):
    def test_CD(self):
        # y = x, this reaffirms if that condition is maintained
        As, Bs, Cs, Ds = state_space_matrices_symmetric(V0, hp0, th0, m)
        Aa, Ba, Ca, Da = state_space_matrices_asymmetric(V0, hp0, m)

        # I am ok with using == here, because the values should be set to these exact values and there shouldn't be any 
        # floating point weirdness going on
        self.assertTrue(np.all(Cs == np.eye(4)))
        self.assertTrue(np.all(Ca == np.eye(4)))
        self.assertTrue(np.all(Ds == np.zeros((4, 1))))
        self.assertTrue(np.all(Da == np.zeros((4, 2))))


    def test_dimensions(self):
        As, Bs, Cs, Ds = state_space_matrices_symmetric(V0, hp0, th0, m)
        Aa, Ba, Ca, Da = state_space_matrices_asymmetric(V0, hp0, m)
        self.assertSequenceEqual(As.shape, (4, 4))
        self.assertSequenceEqual(Aa.shape, (4, 4))
        self.assertSequenceEqual(Bs.shape, (4, 1))
        self.assertSequenceEqual(Ba.shape, (4, 2))
        self.assertSequenceEqual(Cs.shape, (4, 4))
        self.assertSequenceEqual(Ca.shape, (4, 4))
        self.assertSequenceEqual(Ds.shape, (4, 1))
        self.assertSequenceEqual(Da.shape, (4, 2))

    def test_singular(self):
        As, *_ = state_space_matrices_symmetric(V0, hp0, th0, m)
        Aa, *_ = state_space_matrices_asymmetric(V0, hp0, m)

        self.assertNotAlmostEqual(np.linalg.det(As), 0, 3)
        self.assertNotAlmostEqual(np.linalg.det(Aa), 0, 3)

    """def test_bounded(self):
        # Tests if the A-matrix itself remains reasonable bounded
        As, Bs, Cs, Ds = state_space_matrices_symmetric(V0, hp0, th0, m)
        Aa, Ba, Ca, Da = state_space_matrices_asymmetric(V0, hp0, m)

        x = np.zeros((4, 1))
        x = As @ x + Bs @ np.ones([1, 1])
        for i in range(10):
            x = As @ x
        
        self.assertLess(np.linalg.norm(x), 1e6)

        x = np.zeros((4, 1))
        x = Aa @ x + Ba @ np.ones([2, 1])
        for i in range(10):
            x = Aa @ x
        
        self.assertLess(np.linalg.norm(x), 1e6)"""


    def test_known_values_sym(self):
        # c.f. Flight dynamics reader page 111, table 4.9
        rho    = rho0 * pow(1 + lambda_ * hp0 / Temp0, -g / (lambda_*R) + 1)
        W      = m * g
        muc    = m / (rho * S * c)
        CX0    = W * np.sin(th0) / (0.5 * rho * V0 * V0 * S)
        CZ0    = -W * np.cos(th0) / (0.5 * rho * V0 * V0 * S)
        
        A, B, C, D = state_space_matrices_symmetric(V0, hp0, th0, m)

        denom_x = c / V0 * 2 * muc
        self.assertAlmostEqual(A[0, 0], CXu / denom_x)
        self.assertAlmostEqual(A[0, 1], V0 * CXa / denom_x)  # *V
        self.assertAlmostEqual(A[0, 2], V0 * CZ0 / denom_x)  # *V
        self.assertAlmostEqual(A[0, 3], c * CXq / denom_x)    # * c

        denom_z = c / V0 * (2 * muc - CZadot)
        self.assertAlmostEqual(A[1, 0], 1/V0 * CZu / denom_z)  # /V
        self.assertAlmostEqual(A[1, 1], CZa / denom_z)
        self.assertAlmostEqual(A[1, 2], -CX0 / denom_z)
        self.assertAlmostEqual(A[1, 3], c/V0 *  (2*muc + CZq) / denom_z)    # * c/V

        self.assertAlmostEqual(A[2, 0], 0)   # /V
        self.assertAlmostEqual(A[2, 1], 0)
        self.assertAlmostEqual(A[2, 2], 0)
        self.assertAlmostEqual(A[2, 3], 1)    # * c/V

        denom_m = c / V0 * 2 * muc * KY2
        self.assertAlmostEqual(A[3, 0], 1/c * (Cmu + CZu * Cmadot / (2*muc - CZadot)) / denom_m)  # * 1/c
        self.assertAlmostEqual(A[3, 1], V0/c * (Cma + CZa * Cmadot / (2*muc - CZadot)) / denom_m)  # * V/c
        self.assertAlmostEqual(A[3, 2],-V0/c * (CX0 * Cmadot / (2*muc - CZadot)) / denom_m)        # * V/c
        self.assertAlmostEqual(A[3, 3], (Cmq + Cmadot * (2*muc + CZq) / (2*muc - CZadot)) / denom_m)

        self.assertAlmostEqual(B[0, 0], V0 * CXde / denom_x)  # /V
        self.assertAlmostEqual(B[1, 0], CZde / denom_z)
        self.assertAlmostEqual(B[2, 0], 0)
        self.assertAlmostEqual(B[3, 0], V0/c * (Cmde + CZde * Cmadot / (2*muc - CZadot)) / denom_m)  # * V/c


    def test_known_values_asym(self):
        # c.f. Flight dynamics reader page 113, table 4.10
        rho    = rho0 * pow(1 + lambda_ * hp0 / Temp0, -g / (lambda_*R) + 1)
        W      = m * g
        mub    = m / (rho * S * b)
        CL     = 2 * W / (rho * V0 * V0 * S)
        
        A, B, C, D = state_space_matrices_asymmetric(V0, hp0, m)
        det = 4*mub * (KX2*KZ2-KXZ*KXZ) * b/V0

        self.assertAlmostEqual(A[0, 0], V0 / b * CYb / (2*mub))
        self.assertAlmostEqual(A[0, 1], V0 / b * CL / (2*mub))
        self.assertAlmostEqual(A[0, 2], CYp / (4*mub))              # * b/2V
        self.assertAlmostEqual(A[0, 3], (CYr - 4*mub) / (4*mub))    # * b/2V

        self.assertAlmostEqual(A[1, 0], 0)
        self.assertAlmostEqual(A[1, 1], 0)
        self.assertAlmostEqual(A[1, 2], 1)    # * b/2V
        self.assertAlmostEqual(A[1, 3], 0)    # * b/2V

        self.assertAlmostEqual(A[2, 0], 2*V0 / b * (Clb*KZ2 + Cnb*KXZ) / det)    # * 2V/b
        self.assertAlmostEqual(A[2, 1], 0)  # * V/2b
        self.assertAlmostEqual(A[2, 2], (Clp * KZ2 + Cnp*KXZ) / det)
        self.assertAlmostEqual(A[2, 3], (Clr * KZ2 + Cnr*KXZ) / det)

        self.assertAlmostEqual(A[3, 0], 2*V0 / b * (Clb*KXZ + Cnb*KX2) / det)    # * 2V/b
        self.assertAlmostEqual(A[3, 1], 0)  # * V/2b
        self.assertAlmostEqual(A[3, 2], (Clp*KXZ + Cnp*KX2) / det)
        self.assertAlmostEqual(A[3, 3], (Clr*KXZ + Cnr*KX2) / det)

        self.assertAlmostEqual(B[0, 0], V0/b * CYda / (2*mub))
        self.assertAlmostEqual(B[0, 1], V0/b * CYdr / (2*mub))
        
        self.assertAlmostEqual(B[1, 0], 0)
        self.assertAlmostEqual(B[1, 1], 0)

        self.assertAlmostEqual(B[2, 0], 2*V0 / b * (Clda*KZ2+Cnda*KXZ) / det)    # * 2V/b
        self.assertAlmostEqual(B[2, 1], 2*V0 / b * (Cldr*KZ2+Cndr*KXZ) / det)    # * 2V/b

        self.assertAlmostEqual(B[3, 0], 2*V0 / b * (Clda*KXZ+Cnda*KX2) / det)    # * 2V/b
        self.assertAlmostEqual(B[3, 1], 2*V0 / b * (Cldr*KXZ+Cndr*KX2) / det)    # * 2V/b

if __name__ == '__main__':
    unittest.main()
