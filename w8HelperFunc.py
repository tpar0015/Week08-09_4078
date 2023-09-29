# M4 - Autonomous fruit searching
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
sys.path.insert(0, "{}/util".format(os.getcwd()))
from pibot import PenguinPi    # access the robot
import DatasetHandler as dh    # save/load functions
import measure as measure      # measurements
# import pygame                       # python package for GUI

#####################################
'''Import Robot and EKF classes'''
#####################################
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco


########################################################################3
# For path planning


########################################################################3
def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

########################################################################3
def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    print(fname)
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list(sname):
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open(sname, 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    fruit_coor = []

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:

                ''' #BL: added return coordinations of fruits'''
                x_pos = np.round(fruit_true_pos[i][0], 2)
                y_pos = np.round(fruit_true_pos[i][1], 2)
                fruit_coor.append([x_pos, y_pos])
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                    fruit,
                                                    x_pos,
                                                    y_pos))
        n_fruit += 1

    ''' #BL: added return coordinations of fruits'''
    return fruit_coor



# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    # Turn toward waypoint
    robot_angle = np.arctan((waypoint[1]-robot_pose[1])/(waypoint[0]-robot_pose[0])) # rad
    robot_angle = robot_pose[2] - robot_angle

    wheel_vel = 30 # tick
    
    # turn towards the waypoint
    ''' Get baseline'''
    
    dataDir = "{}calibration/param/".format(os.getcwd())
    fileNameB = "{}baseline.txt".format(dataDir)
    # read baseline from numpy formation to float
    baseline = np.loadtxt(fileNameB, delimiter=',')


    turn_time = (baseline * robot_angle) / wheel_vel
    print("Turning for {:.2f} seconds".format(turn_time))

    ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

############################################################################
############################################################################
############        FOR SLAM        #################################### 
############################################################################
############################################################################

'''
    TODO
    - Get a list of Landmarks- used as input for SLAM below
    - Get measure drive from util/measure.py
    - 
    - 
    - check SLAM flag before update???
'''
# get a list of LMS first --> input that into update slam below


def update_slam(self, drive_meas):
    aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)

    # TODO
    lms, aruco_img = aruco_det.detect_marker_positions(self.img)
    
    # CHECKING - pause to print LMS
    if self.request_recover_robot:
        print(lms)
        is_success = self.ekf.recover_from_pause(lms)
        if is_success:
            self.notification = 'Robot pose is successfuly recovered'
            self.ekf_on = True
        else:
            self.notification = 'Recover failed, need >2 landmarks!'
            self.ekf_on = False
        self.request_recover_robot = False
    elif self.ekf_on:  # and not self.debug_flag:
        # Once activate SLAM, state is predicted by measure DRIVE
        # Then being updated with EKF by measure LMS
        self.ekf.predict(drive_meas)
        self.ekf.add_landmarks(lms)
        self.ekf.update(lms)



def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here <----------------



    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################
    # image_poses = {}
    # with open(f'{script_dir}/lab_output/images.txt') as fp:
    #     for line in fp.readlines():
    #         pose_dict = ast.literal_eval(line)
    #         image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # robot_pose = image_poses[image_poses.keys()[-1]]
    ####################################################


    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    print(aruco_true_pos)

    search_list = read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following is only a skeleton code for semi-auto navigation
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue


        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
