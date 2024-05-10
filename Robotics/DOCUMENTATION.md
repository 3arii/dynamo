# `roboticstoolbox` Python Library Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Core Classes](#core-classes)
   - [Robots](#robots)
   - [Kinematics and Dynamics](#kinematics-and-dynamics)
   - [Controllers](#controllers)
   - [Visualization](#visualization)
5. [Advanced Features](#advanced-features)
6. [Examples](#examples)
7. [Reference](#reference)

<a name="introduction"></a>

## 1. Introduction

The `roboticstoolbox` Python library provides a range of tools for robotic simulations, kinematics, dynamics, and control. It allows users to simulate various robotic models, compute forward and inverse kinematics, and apply different controllers. The library is an evolution of the MATLAB-based Robotics Toolbox and offers support for modern Python features.

<a name="installation"></a>

## 2. Installation

To install `roboticstoolbox`, simply use pip:

```bash
pip install roboticstoolbox-python
```

Ensure you also have `spatialmath` installed, which is needed for various mathematical operations:

```bash
pip install spatialmath-python
```

<a name="basic-usage"></a>

## 3. Basic Usage

### Importing the Library

```python
import roboticstoolbox as rtb
import spatialmath as sm
```

### Creating a Robot

```python
# Load the Panda model
panda = rtb.models.Panda()
```

### Forward Kinematics

```python
# Compute the end-effector pose
end_effector_pose = panda.fkine(panda.q)
```

### Inverse Kinematics

```python
# Find the joint angles needed to reach a target pose
target_pose = sm.SE3(0.3, 0.2, 0.3)
joint_angles = panda.ikine(target_pose)
```

<a name="core-classes"></a>

## 4. Core Classes

<a name="robots"></a>

### Robots

- **Class**: `Panda`

  - A robot model representing a Franka Emika Panda robot.
  - **Attributes**:
    - `qlim`: Joint limits.
    - `qr`: Ready configuration.
  - **Methods**:
    - `fkine(q)`: Forward kinematics to compute the end-effector pose given joint angles.
    - `ikine(T)`: Inverse kinematics to compute joint angles for a target pose.

- **Class**: `DHRobot`

  - Implements a Denavit-Hartenberg model of a robot.
  - **Attributes**:
    - `qlim`: Joint limits.
    - `q`: Current joint configuration.
  - **Methods**:
    - `fkine(q)`: Compute forward kinematics.
    - `jacobm(q)`: Manipulability Jacobian matrix.

- **Class**: `SerialLink`
  - Represents a serial-link robot with specified links and joints.

<a name="kinematics-and-dynamics"></a>

### Kinematics and Dynamics

- **Function**: `fkine`

  - Compute forward kinematics for a given set of joint angles.

- **Function**: `ikine`

  - Solve inverse kinematics to find joint angles for a given end-effector pose.

- **Function**: `jacob0`

  - Compute the Jacobian matrix at the base frame.

- **Function**: `jacobe`
  - Compute the Jacobian matrix in the end-effector frame.

<a name="controllers"></a>

### Controllers

- **Function**: `p_servo`
  - Proportional position servo for motion control.

<a name="visualization"></a>

### Visualization

- **Backend**: `swift`

  - Provides 3D visualization using WebGL.

- **Backend**: `PyPlot`
  - Visualization using Matplotlib.

<a name="advanced-features"></a>

## 5. Advanced Features

### Collision Checking

- Built-in support for collision checking with complex shapes.

### Trajectory Planning

- Create and execute smooth trajectories.

### ROS Integration

- Allows for integration with ROS-based robots.

<a name="examples"></a>

## 6. Examples

### Resolved-Rate Motion Control

```python
env = rtb.backends.PyPlot()
env.launch('Panda Resolved-Rate Motion Control Example')

panda = rtb.models.DH.Panda()
panda.q = panda.qr

# Desired end-effector pose
Tep = panda.fkine(panda.q) * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)

arrived = False
env.add(panda)

# Time step
dt = 0.05

while not arrived:
    v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
    panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
    env.step(dt)
```

### Inverse Kinematics Benchmarking

```python
import time
from ansitable import ANSITable

# Set up the solvers
solvers = [
    lambda Tep: rtb.models.Panda().ik_lm_chan(Tep),
    lambda Tep: rtb.models.Panda().ik_lm_wampler(Tep),
    lambda Tep: rtb.models.Panda().ik_lm_sugihara(Tep)
]

# Names of the solvers
solver_names = ["LM Chan", "LM Wampler", "LM Sugihara"]

# Run benchmarking
times = []
for solver in solvers:
    start = time.time()
    for _ in range(100):
        solver(sm.SE3(0.3, 0.2, 0.3))
    total_time = time.time() - start
    times.append(total_time)

# Print the results
table = ANSITable("Method", "Time", border="thin")
for name, t in zip(solver_names, times):
    table.row(name, (t / 100) * 1e6)
table.print()
```

<a name="reference"></a>

## 7. Reference

- [Robotics Toolbox Documentation](https://petercorke.com/toolboxes/robotics-toolbox/)
- [GitHub Repository](https://github.com/petercorke/roboticstoolbox-python)

### Additional Function Descriptions

#### `Tep = panda.fkine(panda.q) * sm.SE3.Trans(velocity_vector[0], velocity_vector[1], velocity_vector[2])`

- **`fkine(q)`**:

  - This is the "Forward Kinematics" function of the robot object.
  - It takes a set of joint angles `q` as input and computes the pose of the end effector.
  - Returns a homogeneous transformation matrix that represents the pose of the end effector in 3D space.

- **`sm.SE3.Trans(x, y, z)`**:
  - This creates a translation transformation matrix along the specified axes.
  - Parameters `x`, `y`, and `z` are translation distances along their respective axes.
  - The multiplication `panda.fkine(...) * sm.SE3.Trans(...)` results in a target pose `Tep`, which shifts the end effector by the specified velocity vector.

#### `v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, gain=1.0, threshold=0.2)`

- **`p_servo`**:
  - This is a proportional position servo controller that computes the required velocity to move the robot towards a desired end-effector pose.
  - Parameters:
    - `current_pose`: The current pose of the robot's end effector.
    - `target_pose`: The desired target pose for the end effector.
    - `gain`: The proportional gain that controls the rate of convergence.
    - `threshold`: Specifies when the servo should consider the robot to have "arrived" at the target.
  - Returns:
    - `v`: The computed velocity vector required to move the end effector towards the target.
    - `arrived`: A boolean indicating whether the end effector has reached the target.

#### `panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v`

- **`jacobe(q)`**:

  - Computes the Jacobian matrix for the end effector, which maps joint velocities to end-effector velocities.
  - Used here to compute the required joint velocities to achieve the desired end-effector velocity.

- **`np.linalg.pinv(...)`**:
  - This is the "pseudo-inverse" function from NumPy, which provides a way to solve the linear system if the Jacobian matrix is non-square or singular.
  - Allows us to compute the best-fit joint velocities that will achieve the desired end-effector velocity `v`.
