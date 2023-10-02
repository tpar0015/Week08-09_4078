
# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import w8HelperFunc as w8
from Prac4_Support.Obstacle import *
import navigate_algo as navi

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# import utility functions
# sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
from gui import GUI             # GUI
import pygame                       # python package for GUI

#####################################
'''Import Robot and EKF classes'''
#####################################
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import shutil

#####################################
from operate_navi_noGui import Operate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", metavar='', type=str, 
                    default='192.168.50.1')
                    # default='localhost')
parser.add_argument("--port", metavar='', type=int, 
                    default=8080)
                    # default=40000)
parser.add_argument("--calib_dir", type=str, default="calibration/param/")
parser.add_argument("--save_data", action='store_true')
parser.add_argument("--play_data", action='store_true')
# parser.add_argument("--map", type=str, default='Home_test_map.txt')
parser.add_argument("--map", type=str, default='map/M4_prac_map_full.txt')
args, _ = parser.parse_known_args()

# read in the true map
fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)

path_navi = True
start = True
slam = True


try:
    if path_navi:
        # create a list start from 1 to 10
        aruco_taglist = [i for i in range(1,11)]

        # print target fruits
        target_fruit_list = w8.read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
        target_fruits_pos = w8.print_target_fruits_pos(target_fruit_list, fruits_list, fruits_true_pos)

        #######################################################################################
        print("\n\t- Setting up params for NAVIGATION - \n")

        # Generate obstacle list based on Selected shape
        # This consists of 10 aruco and 5 obstable fruit
        obstacles, obs_pos = w8.get_obstacles(aruco_true_pos, fruits_true_pos, target_fruits_pos, shape = "rectangle", size = 0.3)

        # #######################################################################################
        print("\n\t- Generating pathway for NAVIGATION - \n")
        waypoint, step_list = w8.get_path(target_fruit_list, target_fruits_pos, obstacles, robot_step_size=0.05, goal_tolerance=0.1 )

        print(f"--> Total steps: {sum(step_list)}")

        print(waypoint)

        # #######################################################################################
        # w8.plot_waypoint(waypoint, target_fruit_list, target_fruits_pos, obs_pos, obstacles)

except KeyboardInterrupt:
    exit()

# point0 = [0, 0]
# point1 = [0.1, 0.1]
# point2 = [0.1, 0.2]
# point3 = [0.1, 0.3]
# point4 = [0.1, 0.4]
# # create waypoint
# waypoint = {
#     "garlic": [point0, point1, point2, point3, point4]
# }

###################################################################################
###################################################################################
#####################         GUI integrated          #############################
###################################################################################
###################################################################################

try:
    if start:
        
        operate = Operate(args, gui = False)
        # operate.stop()
    
        for fruit, path in waypoint.items():

            # Turn off SLAM
            operate.ekf_on = False

            # operate.rotate_360_slam()

            # Ignore first waypoint
            counter_slam = 0
            for waypoint in path[1:]:
                
                ###########################################################
                # 1. Robot drives to the waypoint
                start_pose = operate.get_robot_pose()
                print(f"\nNext waypoint {waypoint}")
                operate.drive_to_point(waypoint)

                ###########################################################
                # 2. Manual compute robot pose (based on start pose & end points)
                operate.manual_set_robot_pose(start_pose, waypoint, debug=False)
                # Debugging
                pose = operate.get_robot_pose()
                theta = np.rad2deg(pose[2])
                print(f"--->Arrived at {waypoint} - Robot pose: {theta}")                    
                counter_slam += 1

                if counter_slam == 12:
                    operate.rotate_360_slam()


            print("--- Stage 2 --- DEBUG --- Reach target fruit")
            input("Enter to continute\n")

            # ###########################################################
            # # 3. When reach each fruit, stop, rotate 360 && localise

            # if slam:
            #     input("Enter to continue\n")
            


except KeyboardInterrupt:
    if start:
        operate.stop()
