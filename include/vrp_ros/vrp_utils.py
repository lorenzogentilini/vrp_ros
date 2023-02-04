import numpy as np
from copy import deepcopy
from pulp import *

class Task:
    def __init__(self,
                 task_type = 'A',
                 task_score = 0.0,
                 x_value = 0.0,
                 y_value = 'A',
                 x_coord = 0.0,
                 y_coord = 0.0):

        self.task_type = task_type
        self.task_score = task_score
        self.x_value = x_value
        self.y_value = y_value
        self.x_coord = x_coord
        self.y_coord = y_coord

class Params:
    def __init__(self,
                 init_pose = np.array([0.0, 0.0, 0.0]),
                 dd_increment_factor = 0.0,
                 max_vv = 0.0,
                 tt_consump = 0.0,
                 land_consump_factor = 0.0,
                 land_time = 0.0,
                 takeoff_consump_factor = 0.0,
                 takeoff_time = 0.0,
                 inspect_consump_factor = 0.0,
                 inspect_time = 0.0):
                 
        self.init_pose = init_pose
        self.dd_increment_factor = dd_increment_factor
        self.max_vv = max_vv
        self.tt_consump = tt_consump
        self.land_consump_factor = land_consump_factor
        self.land_time = land_time
        self.takeoff_consump_factor = takeoff_consump_factor
        self.takeoff_time = takeoff_time
        self.inspect_consump_factor = inspect_consump_factor
        self.inspect_time = inspect_time

# Utils Functions
def convert_data(data):
    for item in data:
        item.x_coord = -(item.x_value - 0.5)
        if item.y_value == 'A':
            item.y_coord = 0.5
        if item.y_value == 'B':
            item.y_coord = 1.5
        if item.y_value == 'C':
            item.y_coord = 2.5
        if item.y_value == 'D':
            item.y_coord = 3.5
        if item.y_value == 'E':
            item.y_coord = 4.5
        if item.y_value == 'F':
            item.y_coord = 5.5
        if item.y_value == 'G':
            item.y_coord = 6.5
        if item.y_value == 'H':
            item.y_coord = 7.5
        if item.y_value == 'I':
            item.y_coord = 8.5
        if item.y_value == 'L':
            item.y_coord = 9.5

def compute_vrp_data(data, params):
    scores = {}
    consump = {}
    distances = {}
    times = {}

    for ii in range(len(data)):
        scores[ii+1] = data[ii].task_score

    for ii in range(len(data)+1):
        consump[ii] = {}
        distances[ii] = {}
        times[ii] = {}

        for jj in range(1, len(data)+1):
            if ii == jj:
                continue

            if ii == 0: # From Initial Condition
                distances[ii][jj] = np.sqrt((params.init_pose[0] - data[jj-1].x_coord)**2 + (params.init_pose[1] - data[jj-1].y_coord)**2)*params.dd_increment_factor  
            else: # From Pose to Pose
                distances[ii][jj] = np.sqrt((data[ii-1].x_coord - data[jj-1].x_coord)**2 + (data[ii-1].y_coord - data[jj-1].y_coord)**2)*params.dd_increment_factor

            # Compute Time
            times[ii][jj] = distances[ii][jj]/params.max_vv

            # Compute Battery Consumption
            consump[ii][jj] = times[ii][jj]*params.tt_consump

            if ii == 0:
                # Case First Position -> To Landing
                if data[jj-1].task_type == 'A':
                    consump[ii][jj] += params.land_time*params.tt_consump*params.land_consump_factor
                    times[ii][jj] += params.land_time

                # Case First Position -> To Inspect
                if data[jj-1].task_type == 'F':
                    consump[ii][jj] += params.inspect_time*params.tt_consump*params.inspect_consump_factor
                    times[ii][jj] += params.inspect_time

            else:
                # Case Landing -> To Landing
                if data[ii-1].task_type == 'A' and data[jj-1].task_type == 'A':
                    consump[ii][jj] += params.takeoff_time*params.tt_consump*params.takeoff_consump_factor
                    times[ii][jj] += params.takeoff_time

                    consump[ii][jj] += params.land_time*params.tt_consump*params.land_consump_factor
                    times[ii][jj] += params.land_time

                # Case Inspect -> To Landing
                if data[ii-1].task_type == 'F' and data[jj-1].task_type == 'A':
                    consump[ii][jj] += params.land_time*params.tt_consump*params.land_consump_factor
                    times[ii][jj] += params.land_time

                # Case Inspect -> To Inspect
                if data[ii-1].task_type == 'F' and data[jj-1].task_type == 'F':
                    consump[ii][jj] += params.inspect_time*params.tt_consump*params.inspect_consump_factor
                    times[ii][jj] += params.inspect_time

                # Case Landing -> To Inspect
                if data[ii-1].task_type == 'A' and data[jj-1].task_type == 'F':
                    consump[ii][jj] += params.takeoff_time*params.tt_consump*params.takeoff_consump_factor
                    times[ii][jj] += params.takeoff_time

                    consump[ii][jj] += params.inspect_time*params.tt_consump*params.inspect_consump_factor
                    times[ii][jj] += params.inspect_time

    return scores, consump, distances, times

# Solve VRP
def solve_vrp(_scores, _consump, _distances, _times, init_battery, kk):
    # Copy Input Quantities
    scores  = deepcopy(_scores)
    consump = deepcopy(_consump)
    distances = deepcopy(_distances)
    times   = deepcopy(_times)

    nn = len(scores)
    scores[nn + 1] = 0 # Zero Score for Final Location

    # Add Zero Consumption for Final Location
    for ii in range(nn+1):
        consump[ii][nn+1] = 0
        times[ii][nn+1] = 0

    # Create Variables x_ij \in {0,1}
    xx = {}
    for ii in range(nn + 1): # i = 0, ..., n
        xx[ii] = {}
        for jj in range(1, nn + 2): # j = 1, ..., n+1
            # Skip if ii = jj
            if ii == jj:
                continue

            xx[ii][jj] = LpVariable('x{}{}'.format(ii,jj), 0, 1, LpBinary)
    
    # Compute Upper Bound on Time & Reward
    T_UB = sum([sum(ti.values()) for ti in times.values()])
    R_UB = sum(scores.values())
    
    # Create Variables \tau_i, \beta_i, r_i
    tt = {}
    bb = {}
    rr = {}

    for ii in range(nn + 2): # i = 0, ..., n+1
        tt[ii] = LpVariable('t{}'.format(ii), 0, T_UB) # 0 <= \tau_i <= \bar{T}

        if ii != 0:
            rr[ii] = LpVariable('r{}'.format(ii), 0, R_UB) # 0 <= r_i <= \bar{S}
            bb[ii] = LpVariable('b{}'.format(ii), 0, init_battery) # 0 <= \beta_i <= B
        else:
            rr[ii] = LpVariable('r{}'.format(ii))
            bb[ii] = LpVariable('b{}'.format(ii))
    
    problem = LpProblem("VRP", LpMinimize)

    # Add Constraints on Initial Battery and Initial Reward
    problem += rr[0] == 0 # r_0 = 0
    problem += bb[0] == init_battery # \beta_0 = B

    # Add Constraint for Initial and Final Location
    problem += sum(xx[0].values()) == 1
    problem += sum(xi[nn+1] for xi in xx.values()) == 1

    # Add Flow Conservation Constraints
    for kk in range(1, nn + 1):
        problem += sum(xi[kk] for xi in xx.values() if kk in xi) == sum(xx[kk].values())

    # Cycle on VRP Graph Edges
    for ii in range(nn + 1): # i = 0, ..., n
        for jj in range(1, nn + 2): # j = 1, ..., n+1

            # Skip non-existent paths
            if jj not in xx[ii]:
                continue

            # Add Precedence Constraint
            M_ij = T_UB + times[ii][jj]
            problem += tt[jj] >= tt[ii] + times[ii][jj] - M_ij*(1 - xx[ii][jj])

            # Add Battery Constraint (1)
            W_ij_up = init_battery - consump[ii][jj]
            problem += bb[jj] >= bb[ii] - consump[ii][jj] - W_ij_up*(1 - xx[ii][jj])

            # Add Battery Constraint (2)
            W_ij_down = init_battery + consump[ii][jj]
            problem += bb[jj] <= bb[ii] - consump[ii][jj] + W_ij_down*(1 - xx[ii][jj])

            # Add Reward Constraint (1)
            Z_j_up = R_UB + scores[jj]
            problem += rr[jj] >= rr[ii] + scores[jj] - Z_j_up*(1 - xx[ii][jj])

            # Add Reward Constraint (2)
            Z_j_down = R_UB - scores[jj]
            problem += rr[jj] <= rr[ii] + scores[jj] + Z_j_down*(1 - xx[ii][jj])

    # Set Cost Function
    # Minimize Distances & Maximize Rewards
    problem += sum(xx[ii][jj]*distances[ii][jj] for ii in range(nn+1) for jj in range(1, nn+1) if jj in xx[ii]) - kk*rr[nn+1]

    # Solve Problem
    problem.solve()

    # Decode Solution
    solution = []
    old_pp = 0 # Initial Position

    while old_pp != nn+1:
        # Find Index of (unique) Entry in xx[old_pp] Equal to 1
        next_pp = [jj for jj in range(1,nn+2) if jj in xx[old_pp] and abs(xx[old_pp][jj].value() - 1) < 1e-5]

        # Append Position to Solution
        old_pp = next_pp[0]

        if old_pp != nn+1:
            solution.append(old_pp)

    reward = float(rr[nn+1].value())
    battery = float(bb[nn+1].value())
    distance = float(sum(xx[ii][jj].value()*distances[ii][jj] for ii in range(nn+1) for jj in range(1,nn+1) if jj in xx[ii]))

    return solution, reward, battery, distance
