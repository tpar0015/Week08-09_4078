
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
from operate_m4_navi import Operate



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
parser.add_argument("--map", type=str, default='map/M4_prac_map_full.txt')
parser.add_argument("--shop", type=str, default='M5_Shopping_list.txt')
parser.add_argument("--plot", type=int, default=1)
args, _ = parser.parse_known_args()

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
        arena = Map((3000,3000), 50, true_map=args.map, shopping_list=args.shop, aruco_size=(500,500), fruit_size=(500, 500))
        arena.generate_map()
        arena.add_aruco_markers()
        arena.add_fruits_as_obstacles()
        arena.get_targets()
        arena.draw_arena(draw_path=True)
        path = arena.get_path_xy()
        
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
        # operate.stop()
        
        operate.prompt_start_slam(aruco_true_pos)

        # for fruit, path in waypoint.items():
        #     # Ignore first waypoint
        #     for waypoint in path[1:]:
        
        waypoint_count = 0

        for one_path in path:
            for waypoint_mm in one_path:
                # Convert to m
                waypoint = []
                for coor in waypoint_mm:
                    waypoint.append(coor * 0.001)

                ###########################################################
                # 1. Robot drives to the waypoint
                # 2. Update robot pose using SLAM (during turn and drive)
                ###########################################################
                print(f"\nNext waypoint {waypoint}")
                operate.drive_to_point(waypoint)
                operate.stop()
                # Debugging
                pose = operate.get_robot_pose()
                x = pose[0]
                y = pose[1]
                theta = np.rad2deg(pose[2])
                print(f"---> ROBOT pose: [{x} {y} {theta}]")
                # input("Enter to continue")

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
