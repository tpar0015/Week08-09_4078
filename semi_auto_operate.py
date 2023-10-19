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
from Navigation_Alternate.mapping import Map

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# import utility functions
# sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import shutil
import argparse
import pygame

#####################################
from operate_m5 import Operate



parser = argparse.ArgumentParser()

# Must have argument
parser.add_argument("map", type=str, help='Full Map')
parser.add_argument("shop", type=str, help='Shopping list')

parser.add_argument("--ip", metavar='', type=str, 
                    default='192.168.50.1')
                    # default='localhost')
parser.add_argument("--port", metavar='', type=int, 
                    default=8080)
                    # default=40000)
parser.add_argument("--calib_dir", metavar='', type=str, default="calibration/param/")
parser.add_argument("--save_data", action='store_true')
parser.add_argument("--play_data", action='store_true')
# For navi
obs_size = 400  # 500
shop_size = 300 # 400
parser.add_argument("--aruco_size", metavar='',  type=int, default=obs_size)
parser.add_argument("--fruit_size", metavar='', type=int, default=obs_size)  
# Entire Robot is within 0.5 from fruit centre
parser.add_argument("--target_size", metavar='',  type=int, default=300)
parser.add_argument("--waypoint_threshold", metavar='', type=int, default=200)
# For control
parser.add_argument("--turn_tick", metavar='', type=int, default=45)
parser.add_argument("--tick", metavar='', type=int, default=60)
parser.add_argument("--unsafe_thres", metavar='', type=int, default=5)
parser.add_argument("--slam_turn_tick", metavar='', type=int, default=15)
# For debug
parser.add_argument("--plot", type=int, default=1)
parser.add_argument("--waypoint_stop", type=int, default=0)
# Fruit detection
parser.add_argument("--yolo", default='latest_model.pt')

args = parser.parse_args()



# read in the true map
fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)
target_fruit_list = w8.read_search_list(args.shop) # change to 'M4_true_shopping_list.txt' for lv2&3
target_fruits_pos = w8.read_target_fruits_pos(target_fruit_list, fruits_list, fruits_true_pos)
##########################################################################################
# operate = Operate(args, gui = False, semi = True)
operate = Operate(args, gui = False, semi = True)
operate.stop()
operate.prompt_start_slam(aruco_true_pos)
# operate.localise_360()
##########################################################################################
waypoint_ctr = 0
try:
    while True:
        right_dist = False
        localise_360_flag = False
        print("#################################\nNew waypoint:")
        operate.print_robot_pose()
        while not right_dist:
            right_input = False
            # get waypoint_x but check if it is float and promt again if not
            while not right_input:
                waypoint_x = input("Enter x in cm: ")
                try:
                    waypoint_x = float(waypoint_x)
                    right_input = True
                except ValueError:
                    print("Please enter a number")

            right_input = False
            # get waypoint_y but check if it is float and promt again if not
            while not right_input:
                waypoint_y = input("Enter y in cm: ")
                try:
                    waypoint_y = float(waypoint_y)
                    right_input = True
                except ValueError:
                    print("Please enter a number")

            cur_pose = operate.get_robot_pose()
            operate.print_robot_pose()
            x = cur_pose[0] * 100
            y = cur_pose[1] * 100
            # theta = cur_pose[2]

            if waypoint_x == 0 and waypoint_y == 0:
                localise_360_flag = True
                right_dist = True
                continue
            # check if the dist to waypoint is < 20cm, if not, prompt everything again using right_dist
            dist = np.sqrt((waypoint_x - x)**2 + (waypoint_y - y)**2)
            print(dist)
            if dist <= 30:
                right_dist = True
            else:
                print("Please enter a waypoint within 20cm of the robot")
                
            

        waypoint = (waypoint_x * 0.01, waypoint_y * 0.01)
        waypoint_ctr += 1

        # Move
        if localise_360_flag:
            tmp = input(" ---- PRESS Y IF YOU WANT TO LOCALISE_360: ")
            if tmp.lower() == "y":
                operate.localise_360()
            else:
                print("Skip the entered waypoint")
        else:
            operate.drive_to_point(waypoint, waypoint_ctr) # always update slam
            operate.stop()

        print(waypoint_ctr)
        if waypoint_ctr >= 6:
            waypoint_ctr = 0

        

        # tmp = input("Press enter to continue to next waypoint, or 'q' to quit: ") 
        # if tmp == 'q':
        #     end = True
    

except KeyboardInterrupt:
    operate.stop()
