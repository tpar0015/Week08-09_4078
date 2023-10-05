# Description: This file contains the controller class for the robot.
import time
import numpy as np
from slam.ekf import EKF
from slam.robot import Robot
from util.pibot import PenguinPi
from gui import GUI

class RobotControl:
    def __init__(self, args):
        ## Robot Parameters
        self.wheel_vel = 20 # Ticks

        self.attempt_num = input("Attempt Number: ")
        self.pose = None

        self.ekf = self.init_ekf(args.datadir, args.ip)
        self.gui = GUI(750, 750, args.map)
        self.pibot = PenguinPi(args.ip)
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
    
    def pose_difference(self, end_pose):
        """Calculate the difference between the current pose and the end pose."""
        pose = self.get_pose()
        angle = np.arctan2(end_pose[1] - pose[1], end_pose[0] - pose[0]) - pose[2]
        dist = np.linalg.norm(end_pose[0:2] - pose[0:2])
        return dist, angle

    def control(self, end_pose, turn_flag):
        """Control the robot to drive to the target."""
        dt = 0.01
        angle_threshold = np.pi/180 # 1 Degree
        distance_threshold = 0.05 # 5 cm
        dist_diff, ang_diff = self.pose_difference(end_pose)
        while (dist_diff < distance_threshold and not turn_flag) or (ang_diff < angle_threshold and turn_flag):
            # Drive
            self.pibot.set_velocity([1, 0])
            dist_diff, ang_diff = self.pose_difference(end_pose)
            self.pibot.set_velocity([0 + 1*(not turn_flag), 1*turn_flag], time=dt)
            # Localize
            self.localize()
            # Update GUI
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
    
    def localize(self):
        pass

    def distance_parameters(self, point):
        pose = self.get_pose()
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
    # Generate a path to each fruit
    pass
