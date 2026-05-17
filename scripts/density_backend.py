import numpy as np
import pandas as pd

# Global seed initialization ensures deterministic simulation across iterative function calls.
np.random.seed(10)

def sim_density(mu, std_dev=5):
    """
    Simulates traffic density using a normal distribution.
    
    Args:
        mu (int/float): Mean traffic density arrival rate.
        std_dev (int/float): Standard deviation of the arrival rate.
        
    Returns:
        int: Simulated vehicle count, clamped to a minimum of 0.
    """
    return max(0, int(np.random.normal(mu, std_dev)))

def cam_density():
    """
    Placeholder for live traffic camera feed integration.
    """
    pass

# Simulation parameters
mu_A = 70
mu_B = 10
mu_C = 15
mu_D = 60
std_dev = 5
alpha = 0.1 # Scaling factor for artificial density distribution

# State initialization
mu_den = {'A': mu_A, 'B': mu_B, 'C': mu_C, 'D': mu_D}
density = {'A': [mu_A], 'B': [mu_B], 'C': [mu_C], 'D': [mu_D]}
ad_density = {'A': [0], 'B': [0], 'C': [0], 'D': [0]}
total = {'A': [mu_A], 'B': [mu_B], 'C': [mu_C], 'D': [mu_D]}
leave_lane = {'A': [0], 'B': [0], 'C': [0], 'D': [0]}

# Main simulation loop
for i in range(100):
    # Calculate the total effective density (actual + artificial) for decision routing
    for lane in total:
        total[lane].append(density[lane][-1] + ad_density[lane][-1])

    # Determine the priority lane for the current cycle based on maximum total density
    cur_max = -1
    cur_lane = None
    for lane in total:
        if total[lane][-1] > cur_max:
            cur_max = total[lane][-1]
            cur_lane = lane

    # Capture state prior to mutation to prevent order-dependent calculation errors
    cleared_density = density[cur_lane][-1]

    # Update state for all lanes based on the routing decision
    for lane in density:
        if lane != cur_lane:
            # Accumulate physical traffic
            new_traffic = sim_density(mu_den[lane], std_dev)
            density[lane].append(density[lane][-1] + new_traffic)
            
            # Distribute artificial density proportionally based on the cleared lane's volume
            increment = cleared_density * alpha
            ad_density[lane].append(ad_density[lane][-1] + increment)
            
            leave_lane[lane].append(0)
        else:
            # Reset state for the routed lane
            density[lane].append(0)
            ad_density[lane].append(0)
            leave_lane[lane].append(1)

# Output final state matrix
print(pd.DataFrame(density))