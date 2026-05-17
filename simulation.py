import numpy as np
import pandas as pd

def sim_density(mu, var=5): # this is used only for simulation purpose
    np.random.seed(10)
    return max(0, int(np.random.normal(mu, var)))

def cam_density(): # the no. of vehicles returned from the traffic cameras
    # this will be the actual implementation (coming from traffic cameras)
    pass

# example no. of vehicles per minute (shows the dissimilarity in lanes)
mu_A = 70
mu_B = 10
mu_C = 15
mu_D = 60
var = 5
alpha = 0.1 # scaling factor to prevent runaway artificial density

mu_den = {'A': mu_A, 'B': mu_B, 'C': mu_C, 'D': mu_D} # example densities
density = {'A': [mu_A], 'B': [mu_B], 'C': [mu_C], 'D': [mu_D]} # initialization
ad_density = {'A': [0], 'B': [0], 'C': [0], 'D': [0]} # initialization
total = {'A': [mu_A], 'B': [mu_B], 'C': [mu_C], 'D': [mu_D]}
leave_lane = {'A': [0], 'B': [0], 'C': [0], 'D': [0]}

for i in range(100): # simulation over 100 rounds
    for lane in total:
        total[lane].append(density[lane][-1] + ad_density[lane][-1]) # lane-wise totals

    cur_max = -1
    cur_lane = None

    for lane in total: # loop to find highest total and find which lane to leave
        if total[lane][-1] > cur_max:
            cur_max = total[lane][-1]
            cur_lane = lane

    for lane in density: 
        if lane != cur_lane: # non cur_lanes
            # density simulation (accumulating cars waiting in line)
            new_traffic = sim_density(mu_den[lane], var)
            density[lane].append(density[lane][-1] + new_traffic)
            
            # artificial density incrementation
            # depends on the density of the lane that just left
            increment = density[cur_lane][-1] * alpha
            ad_density[lane].append(ad_density[lane][-1] + increment)
            
            leave_lane[lane].append(0) # is not a leave lane
        else:
            density[lane].append(0) # density of cur lane becomes 0
            ad_density[lane].append(0) # artificial density of cur lane becomes 0
            leave_lane[lane].append(1) # is a leave lane

print(pd.DataFrame(density))
