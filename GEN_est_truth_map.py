
# This program used to create the estimated truth map
# Using aruco_pose_est_pose.txt and fruit_est_pose.txt

'''
The aruco_pose_est_pose.txt has format
    {"aruco10_0": {"y": -0.6525, "x": -0.5625}, 
    "aruco1_0": {"y": -0.8099999999999999, "x": 1.1849999999999998}, 
    "aruco2_0": {"y": 0.3675, "x": 0.32999999999999996}}

The fruit_est_pose.txt has pretty much the same format:
    {
        "pumpkin_0": {
            "y": 0.8882736268576155,
            "x": 0.020718117689841425
        },
        "potato_0": {
            "y": 0.5126230973833841,
            "x": -0.6602488324194719
        },
        "lemon_0": {
            "y": -0.21756107515764891,
            "x": -0.7425173189095063
        }
    }

The output should have the format:
{"aruco10_0": {"y": -0.6525, "x": -0.5625}, 
"aruco1_0": {"y": -0.8099999999999999, "x": 1.1849999999999998}, 
"aruco2_0": {"y": 0.3675, "x": 0.32999999999999996}, 
"potato_1": {"y": -0.57, "x": -0.075}, 
"garlic_1": {"y": 0.49499999999999994, "x": 1.14}, 
"garlic_0": {"y": 1.095, "x": -1.275}}
'''

import numpy as np
import os

# delete existing est_truth_map.txt
if os.path.exists('est_truth_map.txt'):
    os.remove('est_truth_map.txt')


# add the aruco_est_pose.txt (formatted with comma instead of last "}")
with open('lab_output/aruco_est_pose.txt','r') as f0, open('est_truth_map.txt','a') as f1:      
    for line in f0:
        f1.write(line)

# and add the fruit_est.txt (formatted without first "{")
with open('lab_output/fruit_est_pose.txt','r') as f0, open('est_truth_map.txt','a') as f1:
    for line in f0:
        f1.write(line)


print("\nest_truth_map.txt created\n")
# print out the est_truth_map.txt
with open('est_truth_map.txt','r') as f:
    for line in f:
        print(line, end='')