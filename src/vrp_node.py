#! /usr/bin/env python2.7

import rospy
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import BatteryState
from nav_msgs.msg import Odometry

import numpy as np
from vrp_ros.vrp_utils import Params, Task, convert_data, compute_vrp_data, solve_vrp

class VRP:
    def odom_callback(self, msg):
        # Read Actual Position
        self.pose_valid = True
        self.actual_pose = msg

    def battery_callback(self, msg):
        # Read Actual Battery
        self.battery_valid = True
        self.actual_battery = int(msg.percentage*100 - 5)
        if self.actual_battery < 5:
            self.actual_battery = 0

    def listener_callback(self, msg):
        if(not self.pose_valid):
            print("No Position Received Yet")
            return

        if(not self.battery_valid):
            print("No Battery Level Received Yet")
            return

        try:
            mapToOdom_transform = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0))
        except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("No TF Received Yet")
            return

        maptoBody_pose = do_transform_pose(self.actual_pose.pose, mapToOdom_transform)
        self.params.init_pose = np.array([maptoBody_pose.pose.position.x, maptoBody_pose.pose.position.y, maptoBody_pose.pose.position.z])

        data = []
        for ii in range(0, len(msg.data), 4):
            data.append(Task(task_type = chr(msg.data[ii]), x_value = msg.data[ii+1], y_value = chr(msg.data[ii+2]), task_score = msg.data[ii+3]))
    
        convert_data(data)

        print('Input Data:')
        for item in data:
            print(item.task_type, item.task_score, item.x_value, item.y_value, item.x_coord, item.y_coord)

        scores, consump, distances, times = compute_vrp_data(data, self.params)
        sol, reward, battery, distance = solve_vrp(scores, consump, distances, times, self.actual_battery, self.kk)

        print('Start Position:', self.params.init_pose)
        print('Solution:')
        for item in sol:
            print(data[item-1].task_type, data[item-1].task_score, data[item-1].x_value, data[item-1].y_value, data[item-1].x_coord, data[item-1].y_coord)

        print('Final Reward:', reward)
        print('Final Battery Level:', battery)
        print('Traveled Distance:', distance)

        # Prepare ROS Msg
        msg_out = Int32MultiArray()
        for item in sol:
            msg_out.data.append(ord(data[item-1].task_type))
            msg_out.data.append(data[item-1].x_value)
            msg_out.data.append(ord(data[item-1].y_value))
            msg_out.data.append(data[item-1].task_score)

        self.solution_publisher.publish(msg_out)

    def __init__(self):
        # Attributes
        self.params = Params(np.array([0.0, 0.0, 0.0]),
                             dd_increment_factor = rospy.get_param("~dist_increment", 0.2),
                             max_vv = rospy.get_param("~max_velocity", 0.4),
                             tt_consump = rospy.get_param("~time_battery_consumption", 0.2),
                             land_consump_factor = rospy.get_param("~land_factor", 0.8),
                             land_time = rospy.get_param("~land_time", 4),
                             takeoff_consump_factor = rospy.get_param("~takeoff_factor", 1.2),
                             takeoff_time = rospy.get_param("~takeoff_time", 6),
                             inspect_consump_factor = rospy.get_param("~inspect_factor", 1.0),
                             inspect_time = rospy.get_param("~inspect_time", 30.0))

        self.actual_pose = Odometry()
        self.actual_battery = 0.0
        self.kk = rospy.get_param("~kk", 1000.0)

        self.pose_valid = False
        self.battery_valid = False

        # Subscribers
        self.task_subscriber = rospy.Subscriber("/task_sequence_raw", Int32MultiArray, self.listener_callback)
        self.battery_subscriber = rospy.Subscriber("/mavros/battery", BatteryState, self.battery_callback)
        self.pose_subscriber = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.solution_publisher = rospy.Publisher('/vrp_solution', Int32MultiArray, queue_size=1)    

# Main Function
if __name__ == '__main__':
    rospy.init_node('vrp_node', anonymous = False)
    
    # Initialize Object
    vrp = VRP()

    # Spin
    rospy.spin()
