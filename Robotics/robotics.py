import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import csv

velocity_vector_list = []
filename = "data.csv"

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
example_list = [0.2, 0.2, 0]


def multiple_movement(dt, multiple_velocity_list):
    for velocity_vector in multiple_velocity_list:
        Tep = panda.fkine(panda.q) * sm.SE3.Trans(velocity_vector[0], velocity_vector[1], velocity_vector[2])
        arrived = False

        while not arrived:
            v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, gain=1.0, threshold=0.2)  # Adjusted threshold
            panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
            env.step(dt)

            velocity_vector_list.append(v.tolist())
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)

                # Write each row from the array
                for row in velocity_vector_list:
                    print(row)
                    writer.writerow(row)  
    print(f"Data has been written to {filename}")              


# Task 5 Example
example_list_2 = [[0.2, 0.2, 0] , [0.1, 0.1, 0.45]]
multiple_movement(0.05, example_list_2)

env.hold()