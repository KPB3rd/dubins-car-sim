from math import *
from matplotlib import pyplot as plt

# http://planning.cs.uiuc.edu/node658.html for intro to Dubins car

state = [0, 0, 0] # x y theta
vel = 1
min_tr = 3
goal_heading = 3.14

# TR = 1/tan(maxtheta) -> max_theta = atan(1/min_tr)
max_steering_angle = atan(1./min_tr)

path_x = [0]
path_y = [0]

num_steps = 50
for i in range(num_steps):
    if i == int(num_steps/2): # turn around for funsies
        goal_heading = 0

    # Compute new steering angle (angle of the heading error: goal - current)
    dtheta = atan2(sin(goal_heading - state[2]),cos(goal_heading - state[2]))

    # Enforce turning radius by limiting the steering angle
    clamped_dtheta = min(max_steering_angle, max(-max_steering_angle, dtheta))

    # Update state
    state[0] += vel*cos(state[2])
    state[1] += vel*sin(state[2])
    state[2] += vel*clamped_dtheta

    path_x.append(state[0])
    path_y.append(state[1])

# Plot Path
plt.scatter(path_x, path_y)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
