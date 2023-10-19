# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from datetime import datetime
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
import shutil

#####################################
'''Import Robot and EKF classes'''
#####################################
sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco
from slam_rehaul.ekf_rewrite import EKF
from slam_rehaul.robot import Robot
import slam_rehaul.aruco_detector as aruco
import shutil

from YOLO.detector import Detector



####################################################################################
''' Merge auto_fruit_search in w8 into previous Operate class'''
####################################################################################
class Operate:
    def __init__(self, args, gui=True, semi=False):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port, args.tick, args.turn_tick)
        # 
        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        # self.output = dh.OutputWriter('lab_output')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise SLAM + Driving parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip) # = EKF(Robot)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers
        self.request_recover_robot = False
        self.quit = False
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.notification = '\nPress ENTER to start navigating using SLAM ... \n'
        # Used to computed dt for measure.Drive
        self.control_clock = time.time()
        self.state_log = []

        # Improving slam ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.unsafe_waypoint = 0
        self.unsafe_threshold = args.unsafe_thres
        self.turn_360_vel = args.slam_turn_tick

        # distance away from obstacles
        self.collide_threshold = 0.2
        # distance to go back if about to collide
        self.backward_dist = 0.1
        # number of "about to collide" alerted
        self.collide_ctr = 0
        # threshold for further actions if "being stuck at the obstacles fruit"
        self.collide_ctr_thres = 3
        # this for skip some waypoint from the start as the robot is confident without slam update
        self.waypoint_to_skip_update_slam = 5 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.image_id = 0
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        # initialise images from camera
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector("YOLO/model/" + args.yolo)
            self.yolo_vis = np.ones((240, 320, 3)) * 100

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Use self.command to use self.control func inside POLLING loop
        self.command = {'motion': [0, 0]}
        # self.control_time = 0


        if gui:
            self.gui = GUI(750,750, args.map)
        self.semi_auto = semi

    '''
    ##############################################################################
    ######################      Basic op     #####################################
    ##############################################################################
    '''
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
    - Only do update step if CERTAIN NUM of lms are seen
    '''

    # SLAM with ARUCO markers       
    # def update_slam(self, drive_meas, slam_update_flag=True, weight=0):
    def update_slam(self, drive_meas, waypoint_ctr=100, check_collision = True): # by default, always update slam
        print_period = 0.05
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)

        if self.request_recover_robot: pass
        elif self.ekf_on:
            self.ekf.predict(drive_meas, print_period)
            for lm in lms:
                # Remove those error detection
                if lm.tag not in range(1, 11):
                    lms.remove(lm)
                    continue
                # Debug
                # print(f"lms {lm.tag}", end=" ")
                if not self.semi_auto:
                    # check for collision
                    if lm.dist <= self.collide_threshold:
                        print("!!! Alerted - about to collide with landmarks !!!")
                        self.collide_ctr += 1
                        self.stop()
                        self.manual_backward()
                        return -1

            # skip those first waypoint as the robot is confident without updating slam
            if waypoint_ctr > self.waypoint_to_skip_update_slam:
                self.ekf.update(lms, print_period=print_period)

        return len(lms)

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
        
    def prompt_start_slam(self, aruco_true_pos):
        # Prompting user to start SLAM
        if not self.ekf_on:
            tmp = input(self.notification)
            # if tmp == "s":
            self.ekf_on = True
            self.ekf.init_landmarks(aruco_true_pos)



    '''
    ##############################################################################
    ######################      From M4     ######################################
    ##############################################################################
    '''
    def get_robot_pose(self):
        x = self.ekf.robot.state[0]
        y = self.ekf.robot.state[1]
        theta = self.ekf.robot.state[2]
        self.state_log.append(self.ekf.robot.state)

        return self.ekf.robot.state[0:3, 0]
    
    def print_robot_pose(self):
        pose = self.get_robot_pose()
        x = pose[0]
        y = pose[1]
        theta = np.rad2deg(pose[2])
        if self.semi_auto: #print in cm
            x *= 100
            y *= 100
        print(f"---> ROBOT pose: [{x:.3f} {y:.3f} {theta:.3f}]")

    ####################################################################################
    
    def reach_close_point(self, current_pose_xy, point_xy, threshold = 0.05):
        dist = np.linalg.norm(current_pose_xy[0:2] - point_xy[0:2])
        if dist < threshold:
            return True
        else:
            return False

    def control(self):    
        lv, rv = self.pibot.set_velocity(self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        # running on physical robot (right wheel reversed)
        drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas

    
    def manual_backward(self):
        scale = self.ekf.robot.wheels_scale
        drive_time = self.backward_dist / (scale * self.pibot.tick)
        # set velocity
        self.command['motion'] = [-1, 0]
        # drive
        drive_time += time.time()
        while time.time() <= drive_time:
            self.take_pic()
            drive_meas = self.control()
            self.update_slam(drive_meas, waypoint_ctr = 0) # not updating slam
        self.stop()
        input("Enter to continue")


    def detect_fruit(self, target_fruit_pos):
        if not self.semi_auto:
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)

            dist = []
            for detection in self.detector_output:
                fruit_name = detection[0]
                tmp = self.est_fruit_dist(detection)
                dist.append(tmp)
                # print(f"{fruit_name} is {tmp}m away")
            
            # Check if the robot is close to the target fruit
            # get robot x, y
            reach_target_threshold = 0.4

            # check for collision
            if len(dist) > 0:
                if min(dist) < reach_target_threshold:
                    current_robot_xy = self.get_robot_pose()[:2]
                    if self.reach_close_point(current_robot_xy, target_fruit_pos, threshold = 0.4):
                        print("Hell yeah, reach fruit early")
                        self.stop()
                        input("Enter to continue")
                        return True
                if min(dist) < self.collide_threshold:
                    print(f"!!! Alerted - fruit {min(dist)} !!!")
                    self.collide_ctr += 1
                    self.stop()
                    self.manual_backward()
                    return True
        
        return False

        # For GUI:
        # covert the colour back for display purpose
        # self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)
        # # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
        # self.file_output = (yolo_input_img, self.ekf)

    def est_fruit_dist(self, obj_info):
        focal_length = self.ekf.robot.camera_matrix[0][0]
        target_dimensions_dict = {'Orange': [0.075,0.075,0.073], 'Lemon': [0.08,0.05,0.05], 
                                'Lime': [0.08,0.05,0.05], 'Tomato': [0.07,0.07,0.065], 
                                'Capsicum': [0.095,0.085,0.085], 'Potato': [0.11,0.06,0.062], 
                                'Pumpkin': [0.07,0.085,0.075], 'Garlic': [0.08,0.065,0.075], 


                                'orange': [0.075,0.075,0.073], 'lemon': [0.08,0.05,0.05],
                                'pear': [0,0,0.09], 'berry': [0,0,0.038], 'mango': [0,0,0.057], 
                                'apple': [0,0,0.088], 'pine': [0,0,0.038]}
        
        '''FIND HEIGHT OF OBJECT IN IMAGE'''
        # estimate target pose using bounding box and robot pose
        target_class = obj_info[0]     # get predicted target label of the box
        target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
        true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label
        # compute pose of the target based on bounding box info, true object height, and robot's pose
        pixel_height = target_box[3]
        pixel_center = target_box[0]
        distance = true_height/pixel_height * focal_length  # estimated distance between the object and the robot based on height
        return distance   



    # Rotate 360 to accurately self localise
    def localise_360(self):
        self.stop()
        print("Robot pose before: ")
        self.print_robot_pose()
        print("Do 360 for self-localising")
        # Get params
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width
        # Compute
        tmp = self.pibot.turning_tick
        self.pibot.turning_tick = self.turn_360_vel #for better aruco detection
        turn_360_time = baseline/2 * (2*np.pi) / (scale * self.pibot.turning_tick)
        # Rotate at spot
        self.command['motion'] = [0, 1]
        turn_360_time += time.time()

        while time.time() <= turn_360_time:
            self.take_pic()
            drive_meas = self.control()
            self.update_slam(drive_meas)
        # Reset speed
        self.pibot.turning_tick = tmp
        self.stop()

        print("Just finish 360 turn, current pose: ")
        self.print_robot_pose()
        # input("Done, continue the navi?\n")

    # waypoint_ctr used to skip update slam for the first waypoint
    def drive_to_point(self, waypoint, waypoint_ctr, target_fruit_pos = 0):
        turn_time = self.get_turn_time(waypoint)
        drive_time = self.get_drive_time(waypoint)
        #################################################
        print(f"Turn for {turn_time}")
        if turn_time != 0:
            # Set turn velocity
            if turn_time < 0:
                turn_time *= -1
                self.command['motion'] = [0, -1]
            else:
                self.command['motion'] = [0, 1]
            # Turn at spot
            turn_time += time.time()
            self.control_clock = time.time()
            
            localise_flag = True
            while time.time() <= turn_time:
                self.take_pic()
                drive_meas = self.control()
                # if slam_update_flag:
                lms_detect = self.update_slam(drive_meas, waypoint_ctr)
                if lms_detect == -1 or self.detect_fruit(target_fruit_pos):
                    if self.collide_ctr == self.collide_ctr_thres:
                        self.localise_360()
                        # self.go_around()
                    # skip this waypoint
                    return None
                if lms_detect != 0:
                    localise_flag = False
                
            self.stop()
        # Debug
        # input("Finish turning")
        ##################################################
        print(f"Drive for {drive_time}")
        # Set drive velocity
        self.command['motion'] = [1, 0]
        # Drive straight
        drive_time += time.time()

        while time.time() <= drive_time:
            self.take_pic()
            drive_meas = self.control()
            # if slam_update_flag:
            lms_detect = self.update_slam(drive_meas, waypoint_ctr)
            if lms_detect == -1 or self.detect_fruit(target_fruit_pos):
                if self.collide_ctr == self.collide_ctr_thres:
                    self.localise_360()
                    # self.go_around() # <------------------------------------------------------
                # skip this waypoint
                return None
            if lms_detect != 0:
                localise_flag = False
            # Prevent passing the distance, resulting in TURNING OVER
            if self.reach_close_point(self.get_robot_pose()[:2], waypoint, threshold=0.03):
                break
        self.stop()
        
        ###########################################################################
        # if 0 landmarks detected throughout the drive_to_waypoint
        if localise_flag:
            self.unsafe_waypoint += 1
                
        if self.unsafe_waypoint == self.unsafe_threshold:
            if not self.semi_auto:
                self.localise_360()
                self.unsafe_waypoint = 0


    def get_turn_time(self, waypoint):
        # Get params
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width
        robot_pose = self.get_robot_pose()
        # Angle in WORLD FRAME
        angle_to_waypoint = np.arctan2((waypoint[1]-robot_pose[1]), (waypoint[0]-robot_pose[0]))
        # Angle in ROBOT FRAME
        robot_angle = -robot_pose[2] + angle_to_waypoint
        # Clip
        if (robot_angle > np.pi):
            robot_angle = -2*np.pi + robot_angle
        elif (robot_angle < -np.pi):
            robot_angle = 2*np.pi + robot_angle
        # Debug
        print(f"POINT angle: {np.rad2deg(angle_to_waypoint)}")
        # print(f"ROBOT theta: {np.rad2deg(robot_pose[2])}")
        # print(f"==> turn: {np.rad2deg(robot_angle)}\n")
        # if abs(robot_angle) > np.pi/3:
        #     print("Turning BIG angle !!!!!!!!!!!!!!")
        turn_time = baseline/2 * robot_angle / (scale * self.pibot.turning_tick)
        return turn_time


    def get_drive_time(self, waypoint):
        # Get params
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width
        # Get pose
        robot_pose = self.get_robot_pose()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # after turning, drive straight to the waypoint
        # print(f"DEBUG: {waypoint, robot_pose}")
        robot_dist = ((waypoint[1]-robot_pose[1])**2 + (waypoint[0]-robot_pose[0])**2)**(1/2)
        drive_time = robot_dist / (scale * self.pibot.tick)
        # print(f"Distance to drive: {robot_dist}")
        # print("--------------------------------")
        return drive_time


    def stop(self):
        self.pibot.set_velocity([0, 0])
        self.command['motion'] = [0, 0]

    '''
    ##############################################################################
    ######################      From M2 - Gui     ################################
    ##############################################################################
    '''
    # # save SLAM map
    # def record_data(self):
    #     if self.command['output']:
    #         self.output.write_map(self.ekf)
    #         self.notification = 'Map is saved'
    #         self.command['output'] = False

    # paint the GUI            
    # def draw(self, canvas):
    #     canvas.blit(self.bg, (0, 0))
    #     text_colour = (220, 220, 220)
    #     v_pad = 40
    #     h_pad = 20
    #     # paint SLAM outputs
    #     ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad), not_pause = self.ekf_on)
    #     canvas.blit(ekf_view, (2*h_pad+320, v_pad))
    #     robot_view = cv2.resize(self.aruco_img, (320, 240))
    #     self.draw_pygame_window(canvas, robot_view, 
    #                             position=(h_pad, v_pad)
    #                             )

    #     # canvas.blit(self.gui_mask, (0, 0))
    #     self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad)) # M2
    #     self.put_caption(canvas, caption='Detector (M3)',
    #                      position=(h_pad, 240+2*v_pad)) # M3
    #     self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

    #     pygame.font.init()
    #     TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    #     TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    #     notifiation = TEXT_FONT.render(self.notification,
    #                                       False, text_colour)
    #     canvas.blit(notifiation, (h_pad+10, 596))

    #     time_remain = self.count_down - time.time() + self.start_time
    #     if time_remain > 0:
    #         time_remain = f'Count Down: {time_remain:03.0f}s'
    #     elif int(time_remain)%2 == 0:
    #         time_remain = "Time Is Up !!!"
    #     else:
    #         time_remain = ""
    #     count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
    #     canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
    #     return canvas

    # @staticmethod
    # def draw_pygame_window(canvas, cv2_img, position):
    #     cv2_img = np.rot90(cv2_img)
    #     view = pygame.surfarray.make_surface(cv2_img)
    #     view = pygame.transform.flip(view, True, False)
    #     canvas.blit(view, position)
    
    # @staticmethod
    # def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
    #     caption_surface = TITLE_FONT.render(caption,
    #                                       False, text_colour)
    #     canvas.blit(caption_surface, (position[0], position[1]-25))
        

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
    # fruits_list, fruits_true_pos, aruco_true_pos = w8.read_true_map(args.map)

    # # create a list start from 1 to 10
    # aruco_taglist = [i for i in range(1,11)]

    # # print target fruits
    # target_fruit_list = w8.read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
    # target_fruits_pos = w8.print_target_fruits_pos(target_fruit_list, fruits_list, fruits_true_pos)

    # #######################################################################################
    # print("\n\t- Setting up params for NAVIGATION - \n")

    # # Generate obstacle list based on Selected shape
    # # This consists of 10 aruco and 5 obstable fruit
    # obstacles, obs_pos = w8.get_obstacles(aruco_true_pos, fruits_true_pos, target_fruits_pos, shape = "rectangle", size = 0.3)

    # # #######################################################################################
    # print("\n\t- Generating pathway for NAVIGATION - \n")
    # waypoint, step_list = w8.get_path(target_fruit_list, target_fruits_pos, obstacles, robot_step_size=0.05, goal_tolerance=0.1 )

    # print(f"--> Total steps: {sum(step_list)}")

    ###################################################################################
    ###################################################################################
    #####################         GUI integrated          #############################
    ###################################################################################
    ###################################################################################
    run_start_time = datetime.now()
    operate = Operate(args, gui = False)
    # operate.stop()

    # # #######################################################################################
    # # w8.plot_waypoint(waypoint, target_fruit_list, target_fruits_pos, obs_pos, obstacles)

    # ###################################################################################
    # ###################################################################################
    # #####################         GUI integrated          #############################
    # ###################################################################################
    # ###################################################################################
    # operate = Operate(args, gui = False)
    # # operate.stop()

    # for fruit, path in waypoint.items():

    #     # Turn off SLAM
    #     operate.ekf_on = False

    #     # operate.rotate_360_slam()

    #     # Ignore first waypoint
    #     counter_slam = 0
    #     for waypoint in path[1:]:
            
    #         ###########################################################
    #         # 1. Robot drives to the waypoint
    #         start_pose = operate.get_robot_pose()
    #         print(f"\nNext waypoint {waypoint}")
    #         operate.drive_to_point(waypoint)

    #         ###########################################################
    #         # 2. Manual compute robot pose (based on start pose & end points)
    #         operate.manual_set_robot_pose(start_pose, waypoint, debug=False)
    #         # Debugging
    #         pose = operate.get_robot_pose()
    #         theta = np.rad2deg(pose[2])
    #         print(f"--->Arrived at {waypoint} - Robot pose: {theta}")                    
    #         counter_slam += 1

    #         if counter_slam == 12:
    #             operate.rotate_360_slam()


        print("--- Stage 2 --- DEBUG --- Reach target fruit")
        input("Enter to continute\n")
    operate.state_log = np.array(operate.state_log)
    run_start_time_stamp = run_start_time.strftime("_%Y_%m_%d_%H_%M_%S")
    np.save("slam_log"+run_start_time_stamp+".npy",operate.state_log)
