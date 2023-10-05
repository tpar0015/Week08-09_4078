
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
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco
# import shutil

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
parser.add_argument("--shop", type=str, default="M4_prac_shopping_list.txt")
parser.add_argument("--plot", type=int, default=0)
args, _ = parser.parse_known_args()

# read in the true map
fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)

# if path_navi:
if args.plot:
    ##########################################################################################
    # create a list start from 1 to 10
    aruco_taglist = [i for i in range(1,11)]

    # print target fruits
    target_fruit_list = w8.read_search_list(args.shop) # change to 'M4_true_shopping_list.txt' for lv2&3
    target_fruits_pos = w8.print_target_fruits_pos(target_fruit_list, fruits_list, fruits_true_pos)

    #######################################################################################
    print("\n\t- Setting up params for NAVIGATION - \n")

    # Generate obstacle list based on Selected shape
    # This consists of 10 aruco and 5 obstable fruit
    obstacles, obs_pos = w8.get_obstacles(aruco_true_pos, fruits_true_pos, target_fruits_pos, shape = "rectangle", size = 0.3)

    # #######################################################################################
    print("\n\t- Generating pathway for NAVIGATION - \n")
    waypoint, step_list = w8.get_path(target_fruit_list, target_fruits_pos, obstacles, 
                                        robot_step_size= 0.05, 
                                        goal_tolerance= 0.3)

    print(f"--> Total steps: {sum(step_list)}")
    # print(waypoint)

    # #######################################################################################

    w8.plot_waypoint(waypoint, target_fruit_list, target_fruits_pos, obs_pos, obstacles)


##########################################################################################
operate = Operate(args, gui = False)
##########################################################################################
end = False

try:
    while not end:
        waypoint_x = input("Enter q to quit | Enter x coordinate of waypoint: ")
        waypoint_y = input("Enter q to quit | Enter y coordinate of waypoint: ")
        if waypoint_x == "q" or waypoint_y == "q":
            end = True

        waypoint = (float(waypoint_x), float(waypoint_y))

        start_pose = operate.get_robot_pose()
        operate.drive_to_point(waypoint)
        operate.manual_set_robot_pose(start_pose, waypoint, debug=True)
        
        # Debugging
        pose = operate.get_robot_pose()
        print(f"--->Arrived at {waypoint} - Robot pose: {np.rad2deg(pose[2])}")

        # tmp = input("Press enter to continue to next waypoint, or 'q' to quit: ") 
        # if tmp == 'q':
        #     end = True
    

except KeyboardInterrupt:
    operate.stop()
