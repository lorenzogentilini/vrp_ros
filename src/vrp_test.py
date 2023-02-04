import numpy as np
from vrp_ros.vrp_utils import Params, Task, convert_data, compute_vrp_data, solve_vrp

# VRP Test
# Params for VRP
init_pose = np.array([-2.5, 4.5, 1.0])
init_battery = 60 # 60% of Battery

kk = 1000 # Reward Gain
tt_consump = 0.2 # 0.5% of Battery Every Second of Flight

takeoff_time = 6 # Seconds
land_time = 4 # Seconds
inspect_time = 35 # Seconds

takeoff_consump_factor = 1.2
land_consump_factor = 0.8
inspect_consump_factor = 1.0

max_vv = 0.4
dd_increment_factor = 1.2 # Due to Obstacles

params = Params(init_pose = init_pose, tt_consump = tt_consump, takeoff_time = takeoff_time, land_time = land_time, inspect_time = inspect_time,
                takeoff_consump_factor = takeoff_consump_factor, land_consump_factor = land_consump_factor, inspect_consump_factor = inspect_consump_factor,
                max_vv = max_vv, dd_increment_factor = dd_increment_factor)

# Fill Data With Some Tasks
data = []

t1 = Task()
t1.task_type = 'A'
t1.task_score = 50
t1.x_value = 16
t1.y_value = 'E'
data.append(t1)

t2 = Task()
t2.task_type = 'A'
t2.task_score = 300
t2.x_value = 11
t2.y_value = 'B'
data.append(t2)

t3 = Task()
t3.task_type = 'F'
t3.task_score = 150
t3.x_value = 10
t3.y_value = 'G'
data.append(t3)

t4 = Task()
t4.task_type = 'F'
t4.task_score = 200
t4.x_value = 18
t4.y_value = 'I'
data.append(t4)

t5 = Task()
t5.task_type = 'A'
t5.task_score = 50
t5.x_value = 7
t5.y_value = 'E'
data.append(t5)

t6 = Task()
t6.task_type = 'F'
t6.task_score = 100
t6.x_value = 20
t6.y_value = 'B'
data.append(t6)

# Convert into Coordinates
convert_data(data)

# Print Data for Debug
print('Input Data:')
for item in data:
    print(item.task_type, item.task_score, item.x_value, item.y_value, item.x_coord, item.y_coord)

# Compute VRP Data
scores, consump, distances, times = compute_vrp_data(data, params)

# Solve VRP
sol, reward, battery, distance = solve_vrp(scores, consump, distances, times, init_battery, kk)

# Print Data for Debug
print('Start Position:', init_pose)
print('Solution:')
for item in sol:
    print(data[item-1].task_type, data[item-1].task_score, data[item-1].x_value, data[item-1].y_value, data[item-1].x_coord, data[item-1].y_coord)

print('Final Reward:', reward)
print('Final Battery Level:', battery)
print('Traveled Distance:', distance)