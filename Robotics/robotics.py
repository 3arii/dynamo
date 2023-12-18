import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

panda = rtb.models.Panda()
panda.q = panda.qr


# Task 4: Get the Robot to move at a specific velocity vector
def move_robot(dt, velocity_list):
    Tep = panda.fkine(panda.q) * sm.SE3.Trans(velocity_list[0], velocity_list[1], velocity_list[2])

    arrived = False
    env.add(panda)

    while not arrived:
        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
        panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
        env.step(dt)

# Task 4 Example
example_list = [0.2, 0.2, 0.45]
move_robot(0.05, example_list)

env.hold()