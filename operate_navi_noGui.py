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



####################################################################################
''' Merge auto_fruit_search in w8 into previous Operate class'''
####################################################################################
class Operate:
    def __init__(self, args, gui=True):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)
        # 
        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise SLAM + Driving parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip) # = EKF(Robot)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers
        self.request_recover_robot = False
        self.quit = False
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.notification = '---------------------Press s to start SLAM'
        # Used to computed dt for measure.Drive
        self.control_clock = time.time()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.image_id = 0
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Use self.command to use self.control func inside POLLING loop
        self.command = {'motion': [0, 0]}

        self.turn_vel = 15  
        self.wheel_vel = 30


        if gui:
            self.gui = GUI(750,750, args.map) 

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
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
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
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        new_tag_detected = False

        if self.request_recover_robot:
            pass
            # is_success = self.ekf.recover_from_pause(lms)
            # if is_success:
            #     self.notification = 'Robot pose is successfuly recovered'
            #     self.ekf_on = True
            # else:
            #     self.notification = 'Recover failed, need >2 landmarks!'
            #     self.ekf_on = False
            # self.request_recover_robot = False

        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)

            # self.ekf.add_landmarks(lms) # <------------------------------

            for lm in lms:
                print(f"---- Detect aruco: {lm.tag}, {lm.position} ----")
                # print(type(lms)) # <-- a list!
                # print(type(lm))
                if lm.tag not in range(1, 11):
                    lms.remove(lm)

                ''' BL: too sleepy'''
                new_tag_detected = True
                    
            self.ekf.update(lms)
        return new_tag_detected

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
    
    def rotate_360_slam(self):
        self.stop()

        # Prompting user to start SLAM
        if not self.ekf_on:
            tmp = input(self.notification)
            if tmp == "s":
                self.ekf_on = True
                _, _, aruco_true_pos = w8.read_true_map(args.map)
                self.ekf.init_landmarks(aruco_true_pos)
                print(f"\n\n--- DEBUG --- cur_pose before SLAM: {self.get_robot_pose()}")

        if self.ekf_on:
            scale = self.ekf.robot.wheels_scale
            baseline = self.ekf.robot.wheels_width
            turn_360_time = baseline/2 * (2*np.pi) / (scale * self.turn_vel)

            start_theta = self.get_robot_pose()[-1]
            start_time = time.time()
            dt = 0.1

            # print(f"start_theta: {start_theta}")
            # print(f"cur_theta: {cur_theta}")
            # print( abs(cur_theta - start_theta))

            # new_tag_detected_flag = False
            ##################################################
            # Make sure it rotate for 7s -- try to get away from the starting_theta !!!
            while time.time() - start_time <= turn_360_time:
                # Turn around
                self.pibot.set_velocity([0, 1], time=dt)

                '''BL: copied from M2'''
                self.take_pic()
                drive_meas = self.control()
                self.update_slam(drive_meas)
                
                # update theta
                cur_theta = self.get_robot_pose()[-1]
            # print("------------ Done with timing ---------------")
            # input("Enter to continue")

            # ##################################################
            # # Make sure that it detect new aruco --> cur_theta changed
            # while not new_tag_detected_flag:
            #     # Turn around
            #     self.pibot.set_velocity([0, 1], time=dt)

            #     '''BL: copied from M2'''
            #     self.take_pic()
            #     drive_meas = self.control()
            #     new_tag_detected_flag = self.update_slam(drive_meas)
                
            #     # update theta
            #     cur_theta = self.get_robot_pose()[-1]

            # ##################################################
            # # Keep rotate till it get back to starting theta (under certain threshold)
            # threshold = 0.1
            # while abs(cur_theta - start_theta) >= threshold:
            #     # Turn around
            #     self.pibot.set_velocity([0, 1], time=dt)

            #     '''BL: copied from M2'''
            #     self.take_pic()
            #     drive_meas = self.control()
            #     self.update_slam(drive_meas)

            #     # update theta
            #     cur_theta = self.get_robot_pose()[-1]

            print(f"--- DEBUG --- cur_pose after SLAM: {self.get_robot_pose()}\n-----------------\n\n")
            input("Done SLAM check")


    def drive_to_point(self, waypoint):
            
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width

        # Get pose
        robot_pose = self.get_robot_pose()

        print("-----------------------")
        # print(f"waypoint: {waypoint}")
        # print(f"robot_pose: {robot_pose}")

        # angle_to_waypoint = np.arctan2(waypoint[1]-robot_pose[1]), waypoint[0]-robot_pose[0]))
        angle_to_waypoint = np.arctan2((waypoint[1]-robot_pose[1]), (waypoint[0]-robot_pose[0]))
        
        '''BL: Mathemetically not a correct way to compute robot_angle'''
        robot_angle = -robot_pose[2] + angle_to_waypoint
        
        if robot_angle == np.pi or robot_angle == -np.pi:
            robot_angle = 0
        elif (robot_angle > np.pi):
            robot_angle = -2*np.pi + robot_angle
        elif (robot_angle < -np.pi):
            robot_angle = 2*np.pi + robot_angle

        print(f"angle to waypoint: {np.rad2deg(angle_to_waypoint)}")
        print(f"Angle to turn: {np.rad2deg(robot_angle)}\n")
        if abs(robot_angle) > np.pi/3:
            input("Turning big angle, enter to continue")
        print("--------------------------------")

        turn_time = baseline/2 * robot_angle / (scale * self.turn_vel)

        # if turn_time != 0:
        # Account for negative angle
        if turn_time < 0:
            turn_time *=-1
            turn_vel = self.turn_vel * -1
        else:
            turn_vel = self.turn_vel
        # this similar to self.command['motion'] in prev M
        self.pibot.turning_tick = turn_vel
        self.pibot.set_velocity([0, 1], time=turn_time)    # turn on the spot

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # after turning, drive straight to the waypoint
        robot_dist = ((waypoint[1]-robot_pose[1])**2 + (waypoint[0]-robot_pose[0])**2)**(1/2)
        drive_time = robot_dist / (scale * self.wheel_vel)
        
        # this similar to self.command['motion'] in prev M
        self.pibot.tick= self.wheel_vel
        self.pibot.set_velocity([1, 0], time=drive_time)   # drive straight


    def manual_set_robot_pose(self, start_pose, end_point, debug = False):
        '''
        TODO: 
        - from inputs (starting robot's pose, end point) --> compute and save robot pose at end point
        - !!! FOR NOW !!! theta at endpoint is along the LINE_TO_WAYPOINT
        '''
        if debug:
            print("------------------------\nStarting manual set up")
            print(f"Start pose: {start_pose}")
            print(f"End_point: {end_point}")

        x = end_point[0]
        y = end_point[1]
        # do pretty much the same calculation as in drive_to_point()
        angle_to_waypoint = np.arctan2((end_point[1]-start_pose[1]),(end_point[0]-start_pose[0])) # rad

        # update robot pose when reach end point
        self.ekf.robot.state[0] = x
        self.ekf.robot.state[1] = y
        # self.ekf.robot.state[2] = - start_pose[2] + angle_to_waypoint
        self.ekf.robot.state[2] = angle_to_waypoint

        if debug:
            print("\nDone manual set up !!!!!!!!")
            print(f"Current robot pose: {self.get_robot_pose()}\n----------------")
            input("Enter to continue")
            


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

    ###################################################################################
    ###################################################################################
    #####################         GUI integrated          #############################
    ###################################################################################
    ###################################################################################
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
                
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

    #     exit()


    #     try: 
    #     if start:
    #         i = 1
    #         operate = Operate(args, gui=False)
    #         operate.stop()
    #         # waypoint = [0.0, 0.0]
    #         # operate.gui.add_manual_waypoint(waypoint)
    #         print("\n\n~~~~~~~~~~~~~\nStarting\n~~~~~~~~~~~~~\n\n")
    #         input("Enter to start")

    #         while start:
    #             # waypoints = operate.gui.waypoints
    #             waypoints = waypoint_test['garlic']
    #             waypoint = waypoints[i]

    #             '''1. Robot drives to the waypoint'''
    #             cur_pose = operate.get_robot_pose()
    #             print(f"Current robot pose: {cur_pose}")
    #             operate.drive_to_point_pooling(waypoint)
                
    #             '''2. Manual compute robot pose (based on start pose & end points)'''
    #             operate.manual_set_robot_pose(cur_pose, waypoint)
    #             # print(f"Finished driving to waypoint: {waypoint}; New robot pose: {operate.get_robot_pose()}")
    #             print("\n\n#####################################")
    #             print(f"Reach cur_waypoint, New robot theta: {np.rad2deg(operate.get_robot_pose()[-1])}")
    #             print("#####################################\n\n")

    #             '''Go to next waypoint'''
    #             if i < len(waypoints) - 1:
    #                 i += 1      

    #             '''STOP'''
    #             if i == len(waypoints) - 1:
    #                 start = False

    # except KeyboardInterrupt:
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
