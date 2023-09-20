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



####################################################################################
''' Merge auto_fruit_search in w8 into previous Operate class'''
####################################################################################
class Operate:
    def __init__(self, args):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise SLAM + Driving parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip) # = EKF(Robot)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers
        self.request_recover_robot = False
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.notification = 'Press ENTER to start SLAM'
        # Used to computed dt for measure.Drive
        self.control_clock = time.time()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Use self.command to use self.control func inside POLLING loop
        self.command = {'motion': [0, 0]}


    '''
    ##############################################################################
    ######################      Basic op     #####################################
    ##############################################################################
    '''

    # Wheel control - using util
    '''Update the class measure.Drive with all driving params'''
    #
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(self.command['motion'])
        # if self.data is not None:
        #     self.data.write_keyboard(lv, rv)
        # measure time 
        dt = time.time() - self.control_clock
        # running in sim
        if args.ip == 'localhost':
            drive_meas = measure.Drive(lv, rv, dt)
        # running on physical robot (right wheel reversed)
        else:
            drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas

    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()

        if self.data is not None:
            self.data.write_image(self.img)

    # save raw images taken by the camera
    def save_image(self):
        self.image_id = len(os.listdir(self.folder))
        # print(os.listdir(self.folder))
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'


    ''' 
    ##############################################################################
    ######################      SLAM related    ##################################
    ##############################################################################
    
    - Uncommented the "add_landmarks" in update_slam()
    '''

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        # use arcuco_detector.py to call detect in-build func
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        
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
        # Run SLAM using ekf
        elif self.ekf_on:  # and not self.debug_flag:
            # Once activate SLAM, state is predicted by measure DRIVE
            # Then being updated with EKF by measure LMS
            self.ekf.predict(drive_meas)
            # self.ekf.add_landmarks(lms)       # Dont need to add new landmarks
            self.ekf.update(lms)

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)
    
    # Read in the map data to save into ekf clas
    # - tag list
    # - markers list
    def create_lms(self, tags, markers):
        self.ekf.taglist = tags
        self.ekf.markers = markers


    '''
    ##############################################################################
    ######################      From M4     ######################################
    ##############################################################################

    Modify drive_to_point:
    + Only input waypoint
    - Get pose straight from ekf.bot
    - Use self.pibot to set_velocity
    
    '''    
    def get_robot_pose(self):
        return self.ekf.robot.state[0:3, 0]
    
    # Waypoint navigation
    # the robot automatically drives to a given [x,y] coordinate
    # note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
    # fully automatic navigation:
    # try developing a path-finding algorithm that produces the waypoints automatically
    ''' Added two arguments'''
    def drive_to_point(self, waypoint, debug=False, wheel_vel = 20):
        

        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width

        # Get pose
        robot_pose = self.get_robot_pose()
        if (debug):
            print(f"\n --- DEBUG --> waypoint to go: [{waypoint}]")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Turn toward waypoint

        # Note that the angle here is between LINE_TO_WAYPOINT and y-axis ==> arctan(delta_x / delta_y)
        # angle_to_waypoint = np.arctan((waypoint[1]-robot_pose[1])/( abs(waypoint[0]-robot_pose[0])) ) # rad
        # # if robot is pointing away from waypoint y direction !!!!!!!!!!!!!
        # if (robot_pose[2] < np.pi) and (robot_pose[2] > 0):
        #     robot_angle = robot_pose[2] + np.pi - angle_to_waypoint
        # else:
        #     robot_angle = robot_pose[2] + angle_to_waypoint

        angle_to_waypoint = np.arctan2((waypoint[1]-robot_pose[1]), (waypoint[0]-robot_pose[0])) # using arctan2 ????
        robot_angle = -robot_pose[2] + angle_to_waypoint
        turn_time = baseline/2 * robot_angle / (scale * wheel_vel)

        # read baseline from numpy formation to float
        if debug:
            print(f"\n--- DEBUG --> baseline: {baseline}, angle_to_waypoint: {angle_to_waypoint}, angle to turn: {robot_angle}\n")
            input('Press ENTER to turn')
            print("Turning for {:.5f} seconds".format(turn_time))

        # if turn_time != 0:
        # Account for negative angle
        if turn_time < 0:
            turn_time *=-1
            turn_vel = wheel_vel * -1
        else:
            turn_vel = wheel_vel
        # this similar to self.command['motion'] in prev M
        self.pibot.turning_tick = turn_vel
        self.pibot.set_velocity([0, 1], time=turn_time)    # turn on the spot

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # after turning, drive straight to the waypoint
        robot_dist = ((waypoint[1]-robot_pose[1])**2+(waypoint[0]-robot_pose[0])**2)**(1/2)
        drive_time = robot_dist / (scale * wheel_vel)
        if debug:
            print(f"\n---- DEBUG --> scale: {scale}, dist: {robot_dist}\n")
            input('Press ENTER to drive')
            print("Driving for {:.5f} seconds".format(drive_time))
        
        # this similar to self.command['motion'] in prev M
        self.pibot.tick= wheel_vel
        self.pibot.set_velocity([1, 0], time=drive_time)   # drive straight
        
        ####################################################
        if debug:
            print(f"\n~~~~~~~~~~~~~~~~~~~~~\n")

    def manual_set_robot_pose(self, start_pose, end_point):
        '''
        TODO: 
        - from inputs (starting robot's pose, end point) --> compute and save robot pose at end point
        - !!! FOR NOW !!! theta at endpoint is along the LINE_TO_WAYPOINT
        '''
        x = end_point[0]
        y = end_point[1]
        # # do pretty much the same calculation as in drive_to_point()
        # angle_to_end = np.arctan((end_point[0]-start_pose[0])/(end_point[1]-start_pose[1])) # rad
        angle_to_waypoint = np.arctan2((waypoint[1]-start_pose[1]), (waypoint[0]-start_pose[0])) # using arctan2 ???

        # update robot pose when reach end point
        self.ekf.robot.state[0] = x
        self.ekf.robot.state[1] = y
        self.ekf.robot.state[2] += - start_pose[2] + angle_to_waypoint

    def stop(self):
        self.pibot.set_velocity([0, 0])
        

'''
########################################################################
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
~~~~~~~~~~~~~~      Main                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
########################################################################
'''
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    # parser.add_argument("--map", type=str, default='Home_test_map.txt')
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt')
    args, _ = parser.parse_known_args()

    # read in the true map
    # aruco_true_pos already in 10x2 np array
    fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)
    # print(f"aruco: \n{aruco_true_pos}")
    # print(f"fruit: \n{fruits_true_pos}")
    # print(fruits_list)
    # create a list start from 1 to 10
    aruco_taglist = [i for i in range(1,11)]

    # print target fruits
    search_list = w8.read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
    target_fruits_pos = w8.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    
    try:
        #######################################################################################    
        print("\n\t- Setting up params for NAVIGATION - \n")

        # Combine aruco landmark and remained fruit as obstacles
        obs_pos = np.array(aruco_true_pos)
        for fruit_pos in fruits_true_pos:
            # flag
            same = False
            x, y = fruit_pos
            for target_fruit in target_fruits_pos:
                x_target, y_target = target_fruit
                if (x == x_target) and (y == y_target):
                    same = True; break
            if not same:
                obs_pos = np.append(obs_pos, [fruit_pos], axis = 0)
                # print(f"New: {obs_pos[-1]}")    
        # print(len(obs_pos))

        # Set up obstacles with Rectangles outline
        obstacles = []
        for obs in obs_pos:
            # print(obstacles)
            # obstacles.append(Circle(c_x=obs[0], c_y=obs[1], radius=0.2))
            obstacles.append(Rectangle(center=obs, width=0.2, height=0.2))

        print("Obstacles len: ", len(obstacles))

        # Set up waypoint using format [x,y], in metres
        # waypoint = [0.3,0.4]
        initial_robot_pos = [0,0]       # exclude theta
        robot_step_size = 0.05
        ccw = False

        waypoint_path = np.zeros([1,2])      # init
        step_list = []

        # #######################################################################################
        print("\n\t- Generating pathway for NAVIGATION - \n")
        # Generate path - list of waypoints
        for target in target_fruits_pos:
            goal_pos = target

            # The code below finds the path using bug2 algorithm
            path = navi.bug2_algorithm(goal_pos, initial_robot_pos, robot_step_size, obstacles, ccw, tolerance = 0.02)
            # print(path.shape)

            # Clip the first 3 and last 3 waypoints
            path = path[3:-3]

            # Append to the big list of waypoint
            waypoint_path = np.append(waypoint_path, path, axis = 0)
            # print(waypoint_path.shape)

            # Append to the list of steps
            step = len(path)
            step_list.append(step)

            # update
            initial_robot_pos = goal_pos
            print(f"Reached target fruit {goal_pos} after {len(path)} steps\n")

        print(f"--> Total steps: {len(waypoint_path)}")

        # Plot all the waypoints
        for waypoint in waypoint_path:
            plt.plot(waypoint[0], waypoint[1], 'ro')
        # Plot all the obstacles
        for obs in obs_pos:
            plt.plot(obs[0], obs[1], 'bx')
        # Plot all the target fruit
        for target in target_fruits_pos:
            plt.plot(target[0], target[1], 'go')
        plt.show()


        ########################################################################################
        
        # Initialise
        # operate = Operate(args)
        # operate.stop()
        # start = 1
        
        # # Save aruco position for SLAM / EKF functions
        # # operate.create_lms(tags = aruco_taglist, markers=aruco_true_pos)
        # # # print aruco locs
        # # for idx, tag in enumerate(aruco_taglist):
        # #     print(f"Arcuco{tag}: \t{operate.ekf.markers[idx]}")

        # #######################################################################################

        
        # print("\n\n~~~~~~~~~~~~~\nStarting\n~~~~~~~~~~~~~\n\n")

        
        #     # while start:
        #     step_counter = 0
        #     waypoint_counter = 0
        #     for idx, waypoint in enumerate(waypoint_path[2:]):

        #         '''1. Robot drives to the waypoint'''
        #         start_pose = operate.get_robot_pose()
        #         print(f"\nNext waypoint {waypoint}")
        #         operate.drive_to_point(waypoint, debug = False, wheel_vel=30)

        #         '''2. Manual compute robot pose (based on start pose & end points)'''
        #         operate.manual_set_robot_pose(start_pose, waypoint)
        #         pose = operate.get_robot_pose()
        #         theta = np.rad2deg(pose[2])
        #         pose[2] = theta
        #         print(f"--->Arrived at {waypoint} - Robot pose: {pose}")

        #         waypoint_counter += 1
        #         if (waypoint_counter == step_list[step_counter]):
        #             step_counter +=1
        #             waypoint_counter = 0
        #             print("Reach target fruit")
        #             input("Enter to continute")

        #         '''3. Rotate at spot 360 & use SLAM to localise'''
        #         # rot_360 = True
        #         # operate.ekf_on = True
        #         # scale = operate.ekf.robot.wheels_scale
        #         # baseline = operate.ekf.robot.wheels_width
        #         # # get self.pibot.turning_tick to compute turning time
        #         # rot_360_time = baseline/2 * (2*np.pi) / (scale * operate.pibot.turning_tick)
        #         # start_time = time.time()
        #         # while rot_360:
        #         #     cur_time = time.time()
        #         #     operate.command["motion"] = [0, 1]
        #         #     drive_meas = operate.control()
        #         #     ''' TODO: check here'''
        #         #     # operate.update_slam(drive_meas)
        #         #     # print(operate.ekf.)
        #         #     # check time to exit
        #         #     if (cur_time - start_time) >= rot_360_time:
        #         #         rot_360 = False
        #         #         operate.command["motion"] = [0, 0]
        #         #         print(f"Finished rotating 360 degree; New robot pose: {operate.get_robot_pose()}")
        #         # start = False

    except KeyboardInterrupt:
        # operate.stop()
        exit()

    # if start:
    #     operate.stop()



'''
------------------------------------------------------------------------------
Future TODO:
- Update drive_to_point so that it can fit in while loop
    * set as command line and input TIME into the func
    * use control()

- Upgrade manual_set_robot_pose() when ROBOT cannot see landmarks (cannot localise)
    * Identify the time when it not see and see landmarks
    * Calculate the pose based on the TIME and VEL

- Modify GUI
'''
