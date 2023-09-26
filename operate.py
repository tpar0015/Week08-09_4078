# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import w8HelperFunc as w8

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from pibot import PenguinPi    # access the robot
import DatasetHandler as dh    # save/load functions
import measure as measure      # measurements
from gui import GUI             # GUI
import pygame                       # python package for GUI

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
    def __init__(self, args, gui=True):

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


        self.wheel_vel = 50
        if gui:
            self.gui = GUI(750,750) 

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
        # lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        
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
    def set_robot_pose(self, state):
        self.ekf.robot.state[0:3, 0] = state

    # Waypoint navigation
    # the robot automatically drives to a given [x,y] coordinate
    # note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
    # fully automatic navigation:
    # try developing a path-finding algorithm that produces the waypoints automatically
    def get_turn_params(self, waypoint):
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width
        robot_pose = self.ekf.robot.state[0:3, 0]

        angle_to_waypoint = np.arctan2((waypoint[1]-robot_pose[1]),(waypoint[0]-robot_pose[0])) # rad
        robot_angle = robot_pose[2] - angle_to_waypoint
        print(np.rad2deg(robot_angle))

        turn_time = baseline/2 * robot_angle / (scale * self.wheel_vel)
        turn_vel = self.wheel_vel

        if turn_time < 0:
            turn_time *=-1
            turn_vel = self.wheel_vel * -1


        return turn_vel, turn_time

    def get_dist_params(self, waypoint):
        # robot_pose = self.ekf.robot.state[0:3, 0]
        robot_pose = self.get_robot_pose()
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width

        robot_dist = ((waypoint[1]-robot_pose[1]**2)+(waypoint[0]-robot_pose[0])**2)**(1/2)
        drive_time = robot_dist / (scale * self.wheel_vel)
        print(f"scale: {scale}, dist: {robot_dist}")
        print("Driving for {:.5f} seconds".format(drive_time))
        return drive_time

    def drive_control(self, turn_vel, wheel_vel, vel_time, dt, turn=False):
        time_series = np.arange(0, vel_time, dt)
        #time_gui_update = []
        average_gui_update_time = 0.055
        time_series = np.arange(0, vel_time + average_gui_update_time*len(time_series), dt)
        for t in time_series:
            robot_pose = self.get_robot_pose()
            if turn == True:
                end_theta = robot_pose[2] + dt*turn_vel
                end_point = robot_pose
                end_point[2] = end_theta
            else:
                end_x = robot_pose[0] + dt*wheel_vel*np.cos(robot_pose[2])
                end_y = robot_pose[1] - dt*wheel_vel*np.sin(robot_pose[2])
                end_point = [end_x, end_y, robot_pose[2]]
            self.manual_set_robot_pose(robot_pose,end_point)
            self.pibot.set_velocity([0 + 1*(not turn), 1*turn], time=dt)
            # EKF
            # GUI
            # Time it
            #time_prev = time.time()
            self.gui_update()
            #time_after = time.time()
            #time_gui_update.append(time_after - time_prev)


        #print(f"Average time for GUI update: {np.mean(time_gui_update)}")

    def gui_update(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.gui.add_waypoint()
                pygame.event.clear()
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.gui.update_state(self.get_robot_pose())
        self.gui.draw()


    def drive_to_point(self, waypoint):
        # Get dir
        # path = os.getcwd() + "/"
        # fileS = "{}calibration/param/scale.txt".format(path)
        # scale = np.loadtxt(fileS, delimiter=',')
        # fileB = "{}calibration/param/baseline.txt".format(path)
        # baseline = np.loadtxt(fileB, delimiter=',')

        # Get pose
        robot_pose = self.ekf.robot.state[0:3, 0]
        print(f"\n~~~~~~~~~\nIn function: current robot pose: [{robot_pose}]")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Turn toward waypoint
        turn_vel, turn_time = self.get_turn_params(waypoint)
        self.pibot.turning_tick = turn_vel
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # after turning, drive straight to the waypoint
        drive_time = self.get_dist_params(waypoint)
        self.pibot.tick = self.wheel_vel
        # this similar to self.command['motion'] in prev M
        dt = 0.1
        self.drive_control(turn_vel, self.wheel_vel, turn_time, dt, turn=True)
        self.drive_control(turn_vel, self.wheel_vel, drive_time, dt, turn=False)
        ####################################################
        print(f"Arrived at [{waypoint[0]}, {waypoint[1]}]")

    def manual_set_robot_pose(self, start_pose, end_point):
        '''
        TODO: 
        - from inputs (starting robot's pose, end point) --> compute and save robot pose at end point
        - !!! FOR NOW !!! theta at endpoint is along the LINE_TO_WAYPOINT
        '''
        x = end_point[0]
        y = end_point[1]
        # do pretty much the same calculation as in drive_to_point()
        theta = np.arctan2((end_point[1]-start_pose[1]),(end_point[0]-start_pose[0])) # rad
        # update robot pose
        self.ekf.robot.state[0] = x
        self.ekf.robot.state[1] = y
        print(end_point)
        self.ekf.robot.state[2] = end_point[2]

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
    parser.add_argument("--map", type=str, default='Home_test_map.txt')
    # parser.add_argument("--map", type=str, default='M4_prac_map_full.txt')
    args, _ = parser.parse_known_args()

    # Initialise
    operate = Operate(args)
    operate.stop()
    start = True

    # read in the true map
    # aruco_true_pos already in 10x2 np array
    fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)
    # create a list start from 1 to 10
    aruco_taglist = [i for i in range(1,11)]

    # print target fruits
    # search_list = w8.read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
    # w8.print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    # Save aruco position
    operate.create_lms(tags = aruco_taglist, markers=aruco_true_pos)
    # print aruco locs
    for idx, tag in enumerate(aruco_taglist):
        print(f"Arcuco{tag}: \t{operate.ekf.markers[idx]}")

    #######################################################################################

    # Set up waypoint using format [x,y], in metres
    waypoint = [0.1,0.1]
    operate.gui.add_manual_waypoint(waypoint)
    print("\n\n~~~~~~~~~~~~~\nStarting\n~~~~~~~~~~~~~\n\n")
    # try:
    #     while True:
    #         # gui
    #         operate.set_robot_pose([1,1,0])
    #         operate.gui_update()
    #         print(operate.get_turn_params([2,2,0]))

    try: 
        i = 0
        while start:
            waypoints = operate.gui.waypoints
            waypoint = waypoints[i]
            '''1. Robot drives to the waypoint'''
            cur_pose = operate.get_robot_pose()
            print(f"Current robot pose: {cur_pose}")
            operate.drive_to_point(waypoint)
            '''2. Manual compute robot pose (based on start pose & end points)'''
            operate.manual_set_robot_pose(cur_pose, waypoint)
            print(f"Finished driving to waypoint: {waypoint}; New robot pose: {operate.get_robot_pose()}")

            '''Go to next waypoint'''
            if i < len(waypoints) - 1:
                i += 1      
            '''3. Rotate at spot 360 & use SLAM to localise'''
            # rot_360 = True
            # operate.ekf_on = True
            # scale = operate.ekf.robot.wheels_scale
            # baseline = operate.ekf.robot.wheels_width
            # # get self.pibot.turning_tick to compute turning time
            # rot_360_time = baseline/2 * (2*np.pi) / (scale * operate.pibot.turning_tick)
            # start_time = time.time()

            # while rot_360:
            #     gui.update_state(operate.get_robot_pose())
            #     gui.draw()
            #     cur_time = time.time()
            #     operate.command["motion"] = [0, 1]
            #     drive_meas = operate.control()

            #     ''' TODO: check here'''
            #     # operate.update_slam(drive_meas)
            #     # print(operate.ekf.)
            #     # check time to exit
            #     if (cur_time - start_time) >= rot_360_time:
            #         rot_360 = False
            #         operate.command["motion"] = [0, 0]
            #         print(f"Finished rotating 360 degree; New robot pose: {operate.get_robot_pose()}")

            
            if i == len(waypoints) - 1:
                start = False

    except KeyboardInterrupt:
        operate.stop()



    operate.stop()



'''
TODO:

Current plan:
1. Go to waypoint
2. Self calculate ROBOT pose
3. Rotate to localise
--> Repeat

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
