# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi    # access the robot
import util.DatasetHandler as dh    # save/load functions
import util.measure as measure      # measurements
import pygame                       # python package for GUI
import shutil                       # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import YOLO components 
from YOLO.detector import Detector

class Operate:
    def __init__(self, args):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save images taken during W7 fruit detection
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            pass
            # shutil.rmtree(self.folder)
            # os.makedirs(f"{self.folder}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise SLAM + Driving parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers
        # Others
        self.request_recover_robot = False
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.notification = 'Press ENTER to start SLAM'
        # Used to computed dt for measure.Drive
        self.control_clock = time.time()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save keyboard control sequence, imgs, SLAM - using util/datasetWriter.py to 
        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        # file name
        self.image_id = 0
        # dictionary to store keyboard commands
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # initialise YOLO / detector parameters
        
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        # file name to save inference (detection) with the matching robot pose and detector labels
        self.pred_fname = ''
        self.file_output = None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GUI related parameters
        self.pred_notifier = False 
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        # images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)

        self.bg = pygame.image.load('pics/gui_mask.jpg')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Our customised flag to continous take image over a period of time
        self.flag = False
        ##########################################################################


    '''
    ##############################################################################
    ######################      From M1     ######################################
    ##############################################################################
    '''

    # Wheel control - using util/pibot.py
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if self.data is not None:
            self.data.write_keyboard(lv, rv)
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
    ######################      From M2     ######################################
    ##############################################################################

    #BL Some note here'''
    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        # use arcuco_detector.py --> call detect func
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
        elif self.ekf_on:  # and not self.debug_flag:
            # Once activate SLAM, state is predicted by measure DRIVE
            # Then being updated with EKF by measure LMS
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
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

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                # image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                          self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False


    '''
    ##############################################################################
    ######################      From M3     ######################################
    ##############################################################################
    '''
    
    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # need to convert the colour before passing to YOLO
            yolo_input_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            self.detector_output, self.yolo_vis = self.detector.detect_single_image(yolo_input_img)

            # covert the colour back for display purpose
            self.yolo_vis = cv2.cvtColor(self.yolo_vis, cv2.COLOR_RGB2BGR)

            # self.command['inference'] = False     # uncomment this if you do not want to continuously predict
            self.file_output = (yolo_input_img, self.ekf)

            # self.notification = f'{len(self.detector_output)} target type(s) detected'

    '''
    ##############################################################################
    ######################      From M4     ######################################
    ##############################################################################
    '''

    # Waypoint navigation
    # the robot automatically drives to a given [x,y] coordinate
    # note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
    # fully automatic navigation:
    # try developing a path-finding algorithm that produces the waypoints automatically
    def drive_to_point(self, waypoint, robot_pose):
        # imports camera / wheel calibration parameters 
        # Get dir
        path = os.getcwd() + "/"
        fileS = "{}calibration/param/scale.txt".format(path)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}calibration/param/baseline.txt".format(path)
        baseline = np.loadtxt(fileB, delimiter=',')
        
        ####################################################
        # TODO: replace with your codes to make the robot drive to the waypoint
        # One simple strategy is to first turn on the spot facing the waypoint,
        # then drive straight to the way point

        # Turn toward waypoint
        robot_angle = np.arctan((waypoint[1]-robot_pose[1])/(waypoint[0]-robot_pose[0])) # rad
        robot_angle = robot_pose[2] - robot_angle

        wheel_vel = 30 # tick
        
        # read baseline from numpy formation to float


        turn_time = abs((baseline * robot_angle) / wheel_vel)
        print("Turning for {:.2f} seconds".format(turn_time[0]))

        ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        
        # after turning, drive straight to the waypoint
        drive_time = 0.0 # replace with your calculation
        print("Driving for {:.2f} seconds".format(drive_time))
        ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
        ####################################################

        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    '''
    ##############################################################################
    ######################      Keyboard control    ##############################
    ##############################################################################
    '''
    # keyboard teleoperation, replace with your M1 codes if preferred        
    def update_keyboard(self):
        for event in pygame.event.get():
            ######################################
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1] - 1, -1)
            # stop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ########### M1 codes ###########
            # # drive forward
            # if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            #     # pass # TODO: replace with your code to make the robot drive forward
            #     self.command['motion'] = [1, 0]
            # # drive backward
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            #     # pass # TODO: replace with your code to make the robot drive backward
            #     self.command['motion'] = [-1, 0]
            # # turn left
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
            #     # pass # TODO: replace with your code to make the robot turn left
            #     self.command['motion'] = [0, 1]
            # # drive right
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            #     # pass # TODO: replace with your code to make the robot turn right
            #     self.command['motion'] = [0, -1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            ######################################################################
            # continuously save image per loop_interval
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                self.flag = not self.flag
            ######################################################################
        
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    '''
    ##############################################################################
    ######################      GUI         ######################################
    ##############################################################################
    '''
    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480 + v_pad),
                                            not_pause=self.ekf_on)
        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.yolo_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view,
                                position=(h_pad, 240 + 2 * v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                       False, text_colour)
        canvas.blit(notifiation, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                            False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    # parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    # parser.add_argument("--yolo_model", default='YOLO/model/best_4_Sep.pt')

    # fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    # print(aruco_true_pos)
    # search_list = read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
    # print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    args, _ = parser.parse_known_args()
    ppi = PenguinPi(args.ip,args.port)

    # pygame.font.init()
    # TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    # TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    # width, height = 700, 660
    # canvas = pygame.display.set_mode((width, height))
    # pygame.display.set_caption('ECE4078 2023 Lab')
    # pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    # canvas.fill((0, 0, 0))
    # splash = pygame.image.load('pics/loading.png')
    # pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
    #                  pygame.image.load('pics/8bit/pibot2.png'),
    #                  pygame.image.load('pics/8bit/pibot3.png'),
    #                  pygame.image.load('pics/8bit/pibot4.png'),
    #                  pygame.image.load('pics/8bit/pibot5.png')]
    # pygame.display.update()

    start = False

    # counter = 40
    # while not start:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             start = True
    #     canvas.blit(splash, (0, 0))
    #     x_ = min(counter, 600)
    #     if x_ < 600:
    #         canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
    #         pygame.display.update()
    #         counter += 2

    operate = Operate(args)

    
    # Set up clock
    clock = pygame.time.Clock()

    # Initialize variables
    
    current_time = pygame.time.get_ticks()
    ################################################################################
    loop_interval = 700  # 1000 milliseconds = 1 second --> for capture image
    ################################################################################
    operate.drive_to_point([0.5, 0.5], operate.ekf.robot.state)
    operate.command['motion'] = [0,0]

    while False:
        # operate.update_keyboard()
        # operate.take_pic()
        # drive_meas = operate.control()
        # operate.update_slam(drive_meas)
        # operate.record_data()
        
        operate.drive_to_point([0.5, 0.5], operate.ekf.robot.state)

        print(operate.ekf.robot.state)
        
        
        # Check if 1 second has passed
        if pygame.time.get_ticks() - current_time >= loop_interval:
            current_time = pygame.time.get_ticks()  # Reset current_time

            if operate.flag:    
                # Your loop code here
                operate.command['save_image'] = True
                print(f" {loop_interval} milliseconds has passed.")

        # operate.save_image()
        # operate.detect_target()
        # visualise
        # operate.draw(canvas)
        # pygame.display.update()