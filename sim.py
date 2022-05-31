from math import *
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import time

# http://planning.cs.uiuc.edu/node658.html for intro to Dubins car

class DubinsCar:
    # Add yaw rate limit?
    def __init__(self, name, init_state, vel, max_steering_angle, guidance):
        self.name = name
        self.state = init_state
        self.max_steering_angle = max_steering_angle
        self.vel = vel
        self.guidance = guidance
        self.path = []
        self.is_done_flag = False

    def step(self, dt):
        if self.is_done_flag:
            return

        # print('Stepping',self.name)
        goal_heading_rad = self.guidance()
        # Compute new steering angle (angle of the heading error: goal - current)
        dtheta = atan2(sin(goal_heading_rad - self.state[2]), cos(goal_heading_rad - self.state[2]))

        # Enforce turning radius by limiting the steering angle
        clamped_dtheta = min(self.max_steering_angle, max(-self.max_steering_angle, dtheta))

        # Update state
        self.state[0] += self.vel*cos(self.state[2]) * dt
        self.state[1] += self.vel*sin(self.state[2]) * dt
        self.state[2] += self.vel*clamped_dtheta * dt

        self.path.append([x for x in self.state])

    def vel_vec(self):
        return [self.vel*cos(self.state[2]), self.vel*sin(self.state[2])]

    def is_done(self, other, threshold=50):
        dist = sqrt((other.state[0]-self.state[0])**2 + (other.state[1]-self.state[1])**2)
        results = (self.is_done_flag or threshold > dist) and self.name != 'Target' # gross
        if results:
            self.is_done_flag = True
        return results

    def predict(self, dt):
        return [self.state[0] + self.vel*cos(self.state[2]) * dt, self.state[1] + self.vel*sin(self.state[2]) * dt, self.state[2]]

def pursuit(me, other):
    return atan2(other.state[1]-me.state[1],other.state[0]-me.state[0])

global_zem  = [0,0]
global_zev = [0,0] # TODO revisit this
def genex(me, other, n_gain, impact_angle_rad):
    # Calc ZEM
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1071.6127&rep=rep1&type=pdf
    # https://ndiastorage.blob.core.usgovcloudapi.net/ndia/2006/garm/wednesday/pamadi.pdf
    rel_pos_vec = np.subtract(other.state[:-1],me.state[:-1])
    dist_apart = np.linalg.norm(rel_pos_vec)
    vel_diff = np.subtract(other.vel_vec(),me.vel_vec())
    rel_pos_unit_vec = np.divide(rel_pos_vec, np.linalg.norm(rel_pos_vec))
    closing_rate = np.dot(vel_diff,rel_pos_unit_vec)
    time_to_go = dist_apart / -closing_rate
    agent_future_state = me.predict(time_to_go)
    target_future_state = other.predict(time_to_go)
    zem = np.subtract(target_future_state[:-1],agent_future_state[:-1])

    # Calc ZEV
    goal_vel_vec = [me.vel*cos(impact_angle_rad),me.vel*sin(impact_angle_rad)]
    zev = np.subtract(goal_vel_vec,me.vel_vec())

    # Combine ZEM ZEV using optimal gains
    k1 =  (n_gain+2)*(n_gain+3)
    k2 = -(n_gain+1)*(n_gain+2)
    if n_gain < 0:
        k1 = 1. # TODO hack
        k2 = 0. # TODO hack
    genex_vec = k1*np.array(zem) + k2*time_to_go*np.array(zev)

    global global_zem    # Needed to modify global copy of globvar
    global_zem = k1*np.array(zem)
    global global_zev    # Needed to modify global copy of globvar
    global_zev = k2*time_to_go*np.array(zev)

    return atan2(genex_vec[1],genex_vec[0])

vel = 20
target_vel = 10

dt = 1 # if set too big we may miss
min_tr = 400
max_steering_angle = atan(1./min_tr) # TR = 1/tan(maxtheta) -> max_theta = atan(1/min_tr)

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, (ax1) = plt.subplots(1, 1)
ax1.plot([], [])
# ax2.plot([], [])
impact_times = {}

def animate(i):
    plt.cla()

    goal_impact_angle = 1.57
    # print('goal angle:',goal_impact_angle)

    target = DubinsCar('Target',[10000, 10000, 3.14],target_vel,max_steering_angle, lambda: 3.14)
    pursuit_agent = DubinsCar('Pursuit',[0, 0, 0],vel,max_steering_angle, lambda: pursuit(pursuit_agent, target))
    pn_agent = DubinsCar('PN',[0, 0, 0],vel,max_steering_angle, lambda: genex(pn_agent, target, -1, goal_impact_angle))
    genex0_agent = DubinsCar('Genex0',[0, 0, 0],vel,max_steering_angle, lambda: genex(genex0_agent, target, 0, goal_impact_angle))
    genex1_agent = DubinsCar('Genex1',[0, 0, 0],vel,max_steering_angle, lambda: genex(genex1_agent, target, 1, goal_impact_angle))
    genex2_agent = DubinsCar('Genex2',[0, 0, 0],vel,max_steering_angle, lambda: genex(genex2_agent, target, 2, goal_impact_angle))

    entities = [target, pursuit_agent, pn_agent, genex0_agent, genex1_agent, genex2_agent]

    for incr in range(i):
        not_done = []
        for entity in entities:
            entity.step(dt)
            if not entity.is_done(target):
                not_done.append(entity.name)
            elif entity.name not in impact_times:
                impact_times[entity.name] = i*dt
                print(entity.name, 'done!')

        if len(not_done) == 1 and not_done[0] == 'Target':
            print("Impact times:", impact_times)
            quit()

    for entity in entities:
        scat = ax1.plot([x[0] for x in entity.path],[x[1] for x in entity.path], label=entity.name)

    # Visualize ZEM ZEV for an agent
    # scat = ax2.plot([0,global_zem[0]],[0,global_zem[1]],c='r')
    # scat = ax2.plot([0,global_zev[0]],[0,global_zev[1]],c='b')
    # scat = ax2.plot([0,global_zem[0] + global_zev[0]],[0,global_zem[1] + global_zev[1]],c='k')
    # ax2.set_xlim([-30000, 30000])
    # ax2.set_ylim([-30000, 30000])
    # print('ZEM:', global_zem,'ZEV:', global_zev)

    plt.legend(loc='best')

    return scat,

anim = animation.FuncAnimation(fig, animate, frames=1000, interval=10)
plt.show()
