
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
parser.add_argument("--turn_tick", metavar='', type=int, default=20)
parser.add_argument("--tick", metavar='', type=int, default=40)
parser.add_argument("--unsafe_thres", metavar='', type=int, default=5)
parser.add_argument("--360_turn_tick", metavar='', type=int, default=40)
# For debug
parser.add_argument("--plot", type=int, default=1)
parser.add_argument("--waypoint_stop", type=int, default=0)
# Fruit detection
parser.add_argument("--yolo", default='latest_model.pt')

args = parser.parse_args()

# read in the true map
fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)

# Flag for Operation
bug2_navi = 0
a_star_navi = 1
start = 1

############################################################
# Path planning, use known map to generate list of waypoint
############################################################
try:
    if a_star_navi:
        arena = Map((2700,2700), 50, 
                    true_map=args.map, shopping_list=args.shop, 
                    aruco_size=(args.aruco_size,args.aruco_size), 
                    fruit_size=(args.fruit_size, args.fruit_size), 
                    target_size=(args.target_size, args.target_size),
                    distance_threshold=args.waypoint_threshold,
                    plot=args.plot  # save as image instead of plotting
                    )
        arena.generate_map()
        arena.add_aruco_markers()
        arena.add_fruits_as_obstacles()
        arena.get_targets()
        arena.draw_arena_v2()
        path = arena.get_path_xy()
        # Ignore the first 0.0, 0.0
        path[0] = path[0][1:]
        
        
    if bug2_navi: 
    # create a list start from 1 to 10
        aruco_taglist = [i for i in range(1,11)]

        # print target fruits
        target_fruit_list = w8.read_search_list(args.shop) # change to 'M4_true_shopping_list.txt' for lv2&3
        target_fruits_pos = w8.print_target_fruits_pos(target_fruit_list, fruits_list, fruits_true_pos)

        #######################################################################################
        print("\n\t- Setting up params for NAVIGATION - \n")

        # Generate obstacle list based on Selected shape
        # This consists of 10 aruco and 5 obstable fruit
        obstacles, obs_pos = w8.get_obstacles(aruco_true_pos, fruits_true_pos, target_fruits_pos, 
                                              shape = "rectangle", 
                                              size = 0.35,   # need to account for robot size
                                              )

        # #######################################################################################
        print("\n\t- Generating pathway for NAVIGATION - \n")
        waypoint, step_list = w8.get_path(target_fruit_list, target_fruits_pos, obstacles, 
                                          robot_step_size= 0.1,
                                          goal_tolerance= 0.3,
                                          ccw=False)

        print(f"--> Total steps: {sum(step_list)}")

        # #######################################################################################
        if args.plot:
            w8.plot_waypoint(waypoint, target_fruit_list, target_fruits_pos, obs_pos, obstacles)

except KeyboardInterrupt:
    exit()


############################################################
# Main operation - drive to waypoints with SLAM
############################################################
try:
    if start:
        
        operate = Operate(args, gui = False)
        operate.stop()
        operate.prompt_start_slam(aruco_true_pos)

        waypoint_ctr = 0
        update_slam_flag = False
        # operate.localise_360()
        # Iterate through each path in from navigation planning
        for one_path in path:
            # Iterate through each waypoint of each path to target fruit (in shopping order)
            for waypoint_mm in one_path:
                waypoint_ctr += 1
                # Convert to m
                waypoint = []
                for coor in waypoint_mm:
                    waypoint.append(coor * 0.001)

                ###########################################################
                # 1. Robot drives to the waypoint
                # 2. Update robot pose using SLAM (during turn and drive)
                ###########################################################
                print("###################################")
                print(f"Next waypoint {waypoint}")
                operate.drive_to_point(waypoint, waypoint_ctr)
                operate.stop()
                # Debugging
                if args.waypoint_stop: 
                    operate.print_robot_pose()
                    input("Enter to continue")

            ###########################################################
            # 3. When reach the target, wait and continue
            ###########################################################
            shopping_time = 3
            print_period = 1
            # print(f"Reach {fruit}, wait for {shopping_time}s\n\n\n")
            print("Reached Fruit")
            cur_time = time.time()
            print_time = cur_time
            while time.time() - cur_time < shopping_time:
                # Print every print_period
                if time.time() - print_time > print_period:
                    print(f"Grabbing the fruit - Hopefully not smashing it")
                    print_time = time.time()



except KeyboardInterrupt:
    if start:
        operate.stop()
