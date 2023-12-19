import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

panda = rtb.models.Panda()
panda.q = panda.qr

env.add(panda)

# Task 4: Get the Robot to move at a specific velocity vector
def move_robot(dt, velocity_list, arrived = False):
    Tep = panda.fkine(panda.q) * sm.SE3.Trans(velocity_list[0], velocity_list[1], velocity_list[2])

    while not arrived:
        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
        panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
        env.step(dt)

# Task 4 Example
#example_list = [0.2, 0.2, 0]
#move_robot(0.05, example_list)

# ISSUES: The robot goes to the first specified velocity vector, and makes it way to the second one.
# even after it reached the second one, it keeps on moving and does unpredictable behaviour. After some
# debugging, I have determined that the issue is with the arrived varible as it doesn't change to true
# after the first instance, but I have not been able to fix it.

# Task 5: Get to robot to move to different velocity vectors
def multiple_movement(dt, mulitple_velocity_list):
    for velocity_vector in mulitple_velocity_list:
        Tep = panda.fkine(panda.qr) * sm.SE3.Trans(velocity_vector[0], velocity_vector[1], velocity_vector[2])

        arrived = False

        while not arrived:
            v, arrived = rtb.p_servo(panda.fkine(panda.qr), Tep, 1)
            panda.qd = np.linalg.pinv(panda.jacobe(panda.qr)) @ v
            env.step(dt)

# Task 5 Example
example_list_2 = [[0.2, 0.2, 0], [0.2, 0.2, 0.45]]
multiple_movement(0.05, example_list_2)

env.hold()