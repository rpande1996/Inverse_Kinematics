import matplotlib.pyplot as plt
import sympy as sy
from sympy import nsimplify
from sympy.physics import mechanics as mc
import numpy as np


def s(x):
    return sy.sin(x)


def c(x):
    return sy.cos(x)


pi = np.pi
armTh = mc.dynamicsymbols('armTh')

th, al, a, d, th1, th2, th3, th4, th5, th6, th7, th8, Z1, Z2, Z3, Z4, Z5, Z6, Z7, d1, d3, d4, d5, d7, d8, t, r = sy.symbols(
    'th, al, a, d, th1, th2, th3, th4, th5, th6, th7, th8, Z1, Z2, Z3, Z4, Z5, Z6, Z7, d1, d3, d4, d5, d7, d8, t, r')

r = 100

A = sy.Matrix([[c(th), -s(th) * c(al), s(th) * s(al), a * c(th)],
               [s(th), c(th) * c(al), -c(th) * s(al), a * s(th)],
               [0, s(al), c(al), d],
               [0, 0, 0, 1]])


def calculate_T_matrix(ali, ai, thi, di):
    global A, e
    A_transform = nsimplify(A.subs({al: ali, a: ai, th: thi, d: di}), tolerance=1e-3, rational=True)
    return A_transform


A1 = calculate_T_matrix(-pi / 2, 0, th1, d1)
A2 = calculate_T_matrix(pi / 2, 0, th2, 0)
A3 = calculate_T_matrix(pi / 2, 0, 0, d3)
A4 = calculate_T_matrix(-pi / 2, 0, th4, 0)
A5 = calculate_T_matrix(-pi / 2, 0, th5, d5)
A6 = calculate_T_matrix(pi / 2, 0, th6, 0)
A7 = calculate_T_matrix(0, 0, th7, d7 + d8)

T1 = A1
T2 = (A1 * A2)
T3 = (T2 * A3)
T4 = (T3 * A4)
T5 = (T4 * A5)
T6 = (T5 * A6)
T7 = (T6 * A7)


def ZMatrix_calculator(T1, T2, T4, T5, T6, T7):
    Z1 = T1[:3, 2]
    Z2 = T2[:3, 2]
    Z3 = T4[:3, 2]
    Z4 = T5[:3, 2]
    Z5 = T6[:3, 2]
    Z6 = T7[:3, 2]
    ZMatrix = [Z1, Z2, Z3, Z4, Z5, Z6]
    return ZMatrix


def J_inv_calculator(joint_angle):
    global T7
    th = [th1, th2, th4, th5, th6, th7]
    Th_values = [joint_angle[0], joint_angle[1], 0, joint_angle[2], joint_angle[3], joint_angle[4], joint_angle[5]]

    j = T7[:3, 3]
    ZMatrices = ZMatrix_calculator(T1, T2, T4, T5, T6, T7)

    j1 = j.jacobian(th)

    Z = ZMatrices[0].row_join(
        ZMatrices[1].row_join(ZMatrices[2].row_join(ZMatrices[3].row_join(ZMatrices[4].row_join(ZMatrices[5])))))

    j2 = j1.col_join(Z)
    j2 = change_values(j2, Th_values)
    j2 = np.array(j2).astype(float)

    jInv = np.linalg.pinv(j2)
    return jInv


def xDot(arm_angle_value):
    X_temp = sy.Matrix([[r * c(armTh), 605, 680 + r * s(armTh)]])

    Xdot = np.array(X_temp.diff(t).subs({armTh.diff(t): 2 * pi / 5, armTh: arm_angle_value})).astype(float)
    X = np.zeros((Xdot.shape[0], 2 * Xdot.shape[1]))
    X[:, :3] = Xdot
    return X


def qDot(t_values, arm_angle_value):
    Xdot = xDot(arm_angle_value)
    Qdot = np.dot(J_inv_calculator(t_values), Xdot.T)
    return Qdot


def it_workspace_t_matrix(joint_angle):
    global T7
    tW = T7.subs({th1: joint_angle[0], th2: joint_angle[1],
                  th4: joint_angle[2], th5: joint_angle[3], th6: joint_angle[4], th7: joint_angle[5], d1: 360, d3: 420,
                  d5: 201 + 198.5, d7: 105.5, d8: 100})
    return tW


def change_values(J_updated, theta):
    J_updated = J_updated.subs({th1: theta[0],
                                th2: theta[1], th3: theta[2], th4: theta[3], th5:
                                    theta[4], th6: theta[5], th7: theta[6], d1: 360, d3: 420, d5: 201 + 198.5,
                                d7: 105.5, d8: 100})
    J_updated = nsimplify(J_updated, tolerance=1e-3, rational=True)
    return J_updated


theta2 = [pi / 2, 0, 0, -pi / 2, 0, 0, 0]
ang_vel_init = qDot(theta2, 0)
q = np.zeros((6, 90))
q[:, 0] = np.array([pi / 2, 0, -pi / 2, 0, 0, 0])
q[:, 1] = q[:, 0] + ang_vel_init.T * 0.014 * 4
bot_arm_angle = np.arange(0, 360, 4) + 90
bot_arm_angle = bot_arm_angle * pi / 180
frame_coor = np.zeros((2, 90))
frame_coor[:, 0] = [0, 780]
for i in range(1, 90):
    coor_point = it_workspace_t_matrix(q[:, i])
    coor_point = coor_point[:3, 3].T
    frame_coor[0, i] = coor_point[0]
    frame_coor[1, i] = coor_point[2]
    ang_vel_it = qDot(q[:, i], bot_arm_angle[i])
    if i < 89:
        q[:, i + 1] = q[:, i] + ang_vel_it.T * 0.014 * 4

for i in range(90):
    plt.scatter(frame_coor[0, i], frame_coor[1, i], color='blue')
    plt.pause(0.001)

plt.savefig("../media/output.png")
plt.show()
