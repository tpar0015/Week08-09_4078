# teleoperate the robot, perform SLAM and object detection

import os
import sys
import time
import cv2
import numpy as np

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
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            pass
            # shutil.rmtree(self.folder)
            # os.makedirs(f"{self.folder}")

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        ####################################################################
        self.flag = False
        self.high_speed = False
        self.turn_360_time = 0

        # self.stopUpdateLM = False
        ####################################################################
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector("YOLO/model/" + args.yolo)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
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

    ''' #BL Some note here'''
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

    def slam_360(self):
        # Get params
        scale = self.ekf.robot.wheels_scale
        baseline = self.ekf.robot.wheels_width
        # Compute
        turn_360_time = baseline/2 * (2*np.pi) / (scale * self.pibot.turning_tick)
        print(turn_360_time)
        # Rotate at spot
        self.turn_360_time = turn_360_time + time.time()
        print(self.turn_360_time)
        self.command['motion'] = [0, 1]

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
            # press V to change speed between M2 and M3
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                if self.high_speed:
                    # Switch to LOW speed
                    self.notification = 'Changed speed to LOW SPEED'
                    self.high_speed = False
                    self.command['motion'] = [0, 0]
                    self.pibot.tick = 30
                    self.pibot.turning_tick = 15
                else:
                    # Switch to HIGH speed
                    self.notification = 'Changed speed to HIGH SPEED'
                    self.high_speed = True
                    self.command['motion'] = [0, 0]
                    self.pibot.tick = 50
                    self.pibot.turning_tick = 20
            ######################################################################
        
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                self.command['output'] = True
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # if l is pressed, 
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                self.notification = 'Save offset.txt'
                # get robot pose
                robot_x = self.ekf.robot.state[0]
                robot_y = self.ekf.robot.state[1]
                robot_theta = self.ekf.robot.state[2]
                # calculate offset
                x_offset = -robot_x
                y_offset = -robot_y
                # print(abs(robot_theta))
                if robot_theta < 0:
                    theta_offset = -(robot_theta + 2*np.pi)
                else:
                    theta_offset = -(robot_theta - 2*np.pi)
                # write these 3 value into a offset.txt
                # with open("offset.txt", 'w') as f:
                #     f.write(f"{x_offset}\n")
                #     f.write(f"{y_offset}\n")
                #     f.write(f"{theta_offset}\n")

                np.save("offset.npy", np.array([x_offset, y_offset, theta_offset]))

            #     self.ekf.no_update_when_360 = not self.ekf.no_update_when_360
            #     self.notification = f'no_update_when_360: {self.ekf.no_update_when_360}'

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
                # if self.turn_360_time == 0:
                #     print("start 360")
                #     self.slam_360()
                
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
                self.ekf.lock_map = True
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo", default='latest_model.pt')

    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2023 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    
    # Set up clock
    clock = pygame.time.Clock()

    # Initialize variables
    
    current_time = pygame.time.get_ticks()
    ################################################################################
    loop_interval = 1000  # 1000 milliseconds = 1 second --> for capture image
    ################################################################################

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        
        # Check if 1 second has passed
        if pygame.time.get_ticks() - current_time >= loop_interval:
            current_time = pygame.time.get_ticks()  # Reset current_time

            if operate.flag:    
                # Your loop code here
                operate.command['save_image'] = True
                print(f" {loop_interval} milliseconds has passed.")

        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()