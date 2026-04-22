from Cit_par import *
import numpy as np


def state_space_matrices_symmetric(V0, hp0, th0, m):
    rho    = rho0 * pow(1 + lambda_ * hp0 / Temp0, -g / (lambda_*R) + 1)
    W      = m * g            # [N]       (aircraft weight)
    muc    = m / (rho * S * c)
    CX0    = W * np.sin(th0) / (0.5 * rho * V0 * V0 * S)
    CZ0    = -W * np.cos(th0) / (0.5 * rho * V0 * V0 * S)

    C1 = np.array([
        [-2 * muc,                0,  0,                  0],
        [       0, (CZadot - 2*muc),  0,                  0],
        [       0,                0, -1,                  0],
        [       0,           Cmadot,  0, -2 * muc * KY2]
    ])
    C2 = np.array([
        [CXu, CXa, CZ0,           CXq],
        [CZu, CZa,-CX0, (CZq + 2*muc)],
        [  0,   0,   0,             1],
        [Cmu, Cma,   0,           Cmq]
    ])
    C3 = np.array([
        [CXde],
        [CZde],
        [   0],
        [Cmde]
    ])
    C1 *= c/V0
    C1[..., 3] *= c/V0
    C2[..., 3] *= c/V0
    C1[..., 0] *= 1/V0
    C2[..., 0] *= 1/V0
    A = -np.linalg.inv(C1) @ C2
    B = -np.linalg.inv(C1) @ C3
    C = np.eye(4)
    D = np.zeros((4, 1))
    return A, B, C, D


def short_period_sym_eigenvalues(A):
    #remove rows and collums due to assumptions
    A = np.delete(A,0,axis=0)
    A = np.delete(A,0,axis = 1)
    A = np.delete(A,1,axis = 1)
    A = np.delete(A,1,axis = 0)

    # calculate eigenvalues
    eigvals = np.linalg.eigvals(A)
    return eigvals


def state_space_matrices_asymmetric(V0, hp0, m):
    rho    = rho0 * pow(1 + lambda_ * hp0 / Temp0, -g / (lambda_*R) + 1)
    W      = m * g
    mub    = m / (rho * S * b)
    CL     = 2 * W / (rho * V0 * V0 * S)

    C1 = np.zeros((4, 4))
    C1[0:2, 0:2] = np.diag([CYbdot - 2 * mub, -.5])
    C1[2:4, 2:4] = 4 * mub * np.array([
        [-KX2, KXZ],
        [KXZ, -KZ2]
    ])
    C1[3, 0] = Cnbdot
    C2 = np.array([
        [CYb, CL, CYp, CYr - 4*mub],
        [  0,  0,   1,           0],
        [Clb,  0, Clp,         Clr],
        [Cnb,  0, Cnp,         Cnr]
    ])
    C3 = np.array([
        [CYda, CYdr],
        [   0,    0],
        [Clda, Cldr],
        [Cnda, Cndr]
    ])


    C1 *= b/V0               # from D_b
    C1[..., 2:4] *= b/V0*.5  # from unit normalization
    C2[..., 2:4] *= b/V0*.5  # from unit normalization

    A = -np.linalg.inv(C1) @ C2
    B = -np.linalg.inv(C1) @ C3
    C = np.eye(4)
    D = np.zeros((4, 2))
    return A, B, C, D

"""
def lowkey_validate_asym():
    m    = 5891.172453984284
    V0   = 101.24371723500197
    h0   = 4562.61164705115

    x = np.array([[
        0.025,
       -0.009233843642539137,
        0.00020585542314573816,
        0.0002769362914251974
    ]]).transpose()

    dxdt = np.array([[
        0,
        0.0003546974640628271,
        0.00029769136567413734,
        0.00023506377554118117
    ]]).transpose()

    u = np.array([[
       -0.007045303952589774,
       -0.03419932367500008
    ]]).transpose()

    #state_space_matrices_symmetric(80, 0, 0, 6000)
    C1, C2, C3 = state_space_matrices_asymmetric(V0, h0, m)


    print(C1 @ dxdt)
    print(C2 @ x)
    print(C3 @ u)

    A = -np.linalg.inv(C1) @ C2
    B = -np.linalg.inv(C1) @ C3
"""