import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import pyLasaDataset as lasa
import data
from scipy.interpolate import interp1d
import time

# Define the vector field function f(x)
def vector_field(x):
    # Example vector fieldx
    # Modify this function as per the desired vector field
    y = np.array([np.sin(x[1]), np.cos(x[0]), 0])  # third dimension is constant
    return y

# Control loop
def follow_vector_field(steps, dt, gain=0.5, threshold=0.05):
    for _ in range(steps):
        # Current end-effector position
        x_current = panda.fkine(panda.q).t

        # Calculate the vector field at the current position and normalize/scale it
        v_field = vector_field(x_current)
        v_field = v_field / np.linalg.norm(v_field) * 0.1  # Example scaling

        # Target pose based on the vector field
        Tep = sm.SE3.Trans(v_field[0], v_field[1], v_field[2]) * panda.fkine(panda.q)

        # Move the robot with adjusted gain and threshold
        v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, gain=gain, threshold=threshold)
        if not arrived:
            panda.qd = np.clip(np.linalg.pinv(panda.jacobe(panda.q)) @ v, -1, 1)  # Limiting joint velocities
        else:
            break

        # Step the simulation
        env.step(dt)

'''
lasa_data = getattr(lasa.DataSet, dataset_name)
demos = lasa_data.demos
pos = np.array([demo.pos for demo in demos])[0]  # Using the first demonstration
pos = pos[:, start:end]  # Select a portion of the trajectory
'''

def follow_lasa_trajectory(dataset_name, start=15, end=None, dt=0.05, initial_gain=0.1, final_gain=0.02, z_offset=0.5, interp_factor=5, position_tolerance=0.05, time_limit=15):
    assert dataset_name in data.data_set_names
    lasa_data = getattr(lasa.DataSet, dataset_name)
    demos = lasa_data.demos
    pos = np.array([demo.pos for demo in demos])[0]
    pos = pos[:, start:end]

    x = np.linspace(0, 1, pos.shape[1])
    x_interp = np.linspace(0, 1, pos.shape[1] * interp_factor)
    interp_func = interp1d(x, pos, kind='cubic', axis=1)
    pos_interp = interp_func(x_interp)

    pos_3d = np.vstack((pos_interp, np.ones((1, pos_interp.shape[1])) * z_offset))

    initial_distance = np.linalg.norm(panda.fkine(panda.q).t - pos_3d[:, 0])
    final_point = pos_3d[:, -1]

    for p in pos_3d.T:
        Tep = sm.SE3(p[0], p[1], p[2])
        start_time = time.time()
        arrived = False

        while arrived == False:
            current_distance = np.linalg.norm(panda.fkine(panda.q).t - p)
            endpoint_distance = np.linalg.norm(panda.fkine(panda.q).t - final_point)

            if endpoint_distance > 0.1:
                gain = initial_gain * (current_distance / initial_distance)
            else:
                gain = final_gain

            damping = max(0.1, 1 - current_distance / initial_distance)

            v, _ = rtb.p_servo(panda.fkine(panda.q), Tep, gain=gain, threshold=0.05)
            panda.qd = np.clip(np.linalg.pinv(panda.jacobe(panda.q)) @ v, -damping, damping)
            env.step(dt)

            if (time.time() - start_time) > time_limit:
                return 0

            if np.linalg.norm(panda.fkine(panda.q).t - p) < position_tolerance:
                arrived = True

env = swift.Swift()
env.launch(realtime=True)
panda = rtb.models.Panda()
panda.q = panda.qr
env.add(panda)

follow_lasa_trajectory('Trapezoid')

env.hold()
