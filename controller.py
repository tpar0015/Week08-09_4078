# Description: This file contains the controller class for the robot.
import time
import numpy as np
from slam.ekf import EKF
from slam.robot import Robot
from util.pibot import PenguinPi
import util.measure as measure
from util.gui import GUI
class RobotControl:
    def __init__(self, args):
        ## Robot Parameters
        self.wheel_vel = 20 # Ticks

        self.attempt_num = input("Attempt Number: ")
        self.pose = [0,0,0]

        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        # self.gui = GUI(750, 750, args.map)
        self.pibot = PenguinPi(args.ip, args.port)
        self.scale = self.ekf.robot.wheels_scale 
        self.baseline = self.ekf.robot.wheels_width

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

    def get_pose(self):
        return self.pose
    def set_pose(self):
        self.pose = self.ekf.robot.state
    
    def pose_difference(self, end_pose):
        """Calculate the difference between the current pose and the end pose."""
        pose = self.get_pose()
        angle = np.arctan2(end_pose[1] - pose[1], end_pose[0] - pose[0]) - pose[2]
        dist = np.linalg.norm(np.array(end_pose[0:2]) - np.array(pose[0:2]))
        return dist, angle

    def drive_to_point(self, point):
        turn_time = self.angle_parameters(point)
        if turn_time > 0:
            turn_flag = 1
        elif turn_time < 0:
            turn_flag = -1
        # Turn
        self.control(point, turn_flag)
        # Drive
        self.control(point, 0)

    def control(self, end_pose, turn_flag):
        """Control the robot to drive to the target. Drives until robot_pose 
        is within 5 cm and 1 degree of the target pose."""
        dt = 0.01
        angle_threshold = np.pi/180 # 1 Degree
        distance_threshold = 0.05 # 5 cm
        dist_diff, ang_diff = self.pose_difference(end_pose)

        while (dist_diff < distance_threshold and not turn_flag) or (ang_diff < angle_threshold and turn_flag):
            # Drive
            lv, rv = self.pibot.set_velocity([0 + 1*(not turn_flag), 1*turn_flag])
            drive_meas = measure.Drive(lv, -rv, dt)
            # Localize
            self.localize(drive_meas)
            # Update GUI
            # self.update_gui() 
            self.localize()
            # Update GUIv
            self.update_gui() 
            # Update loop conditions
            dist_diff, ang_diff = self.pose_difference(end_pose)

    def generate_maps(self):
        """Drive around to generate a map of the aruco markers."""
        # SLAM Map

        # Fruit Classification

        # Target Pose Estimation
        pass

    def classify_fruits(self):
        """Classify the fruits in the map."""
        pass
    
    def localize(self, drive_meas):
        # Update Slam
        self.ekf.predict(drive_meas)
        self.ekf.update(drive_meas)
        self.set_pose()
        


    def distance_parameters(self, point):
        pose = self.get_pose()
        dist = np.linalg.norm
        dist = np.linalg.norm(point - pose[0:2])
        # Driving Time
        drive_time = dist / (self.scale * self.wheel_vel) 
        return drive_time

    def angle_parameters(self, point):
        pose = self.get_pose()
        angle = np.arctan2(point[1] - pose[1], point[0] - pose[0]) - pose[2]
        # Turn Time
        turn_time = self.baseline/2 * angle / (self.scale * self.wheel_vel)
        return turn_time

    def update_gui(self):
        pass

if __name__ == "__main__":
    # Drive around to generate a map of the aruco markers and fruits
    # Target Pose Estimation
    # Read in target fruits 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--map", type=str, default='M4_slam_test.txt')

    args, _ = parser.parse_known_args()
    robot = RobotControl(args)
    robot.drive_to_point([1,1,0])
