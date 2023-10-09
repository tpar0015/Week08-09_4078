# Version 4
# Rewrite to my understanding
# Modified by Bill
# Oct 5

import numpy as np
from mapping_utils import MappingUtils
import cv2
import math
import pygame

import pandas as pd

'''
Summarise from textbook chap 6 - localisation

    1. Using a map (known locations of landmarks)
        - calculate distance and (bearing angle) + COVAR for measurement_error
        * find INNOVATION (error between measurement and prediction from robot_state)
        - Linearise (if needed) the measurement function
        * Apply EKF steps

        
        - further note:
            * generate plot to check error (innovation) overtime?
                --> should "not grow monotonically overtimes"
            * sensor fusing
            * in practice, unknown identity of landmark, Particle KF is useful use hypothesis to choose detected landmarks
'''


class EKF:
    # Implementation of an EKF for SLAM
    # The state is ordered as [x; y; theta; l1x; l1y; ...; lnx; lny]

    ##########################################
    # Utility
    # Add outlier rejection here
    ##########################################

    def __init__(self, robot):

        # Lock the aruco poses as it were given
        self.lock_map = True

        ###################################
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        self.taglist = []
        
        # If using known map
        num_landmarks = 10
        self.state_num = 23     # set manually for now, this include robot pose and arcuo pose
        
        ###################################
        # Covariance matrix
        self.P = np.eye(self.state_num)*1e3
        #self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3

        ###################################
        # GUI
        self.robot_init_state = None
        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./pics/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./pics/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./pics/8bit/pibot_top.png')

        ###################################
        ## setup logs to be used for monitoring and in algoritms

        # log Innovation v = z - z_hat
        self.v = []
        # keep track of NIS, Normalized Innovation Squared
        self.nis = []

        ### chi-squared Distribution <-------------------- *Question*
        ## would require scipy

        # Create some sort of pose uncertainty logging
        # self.pose_uncertainty = []

        
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    ##########################################
    # From lecture note / slides:
    # - F     : Jacobian of model
    # - Q     : Covariance of model
    # - P     : Covariance of state (including model and landmarks)
    # 
    # - z     : measurement --> return landmark position
    # - R     : Covariance of measurement
    # - H     : Jacobian of measurement (multiply -1)
    
    # ########################################

    '''
    Input:
        - drive measure, which has:
            * drive time
            * wheel vel 
            * wheel VARIANCE (fixed at 1 for now)
    Output ==> self update
        - P:
        - robot.drive = manual_set_robot_pose
    Description:
        - This use the raw_drive_meas to drive, manual set pose, and update state covar
    '''
    def predict(self, raw_drive_meas):
        
        # Jacobian of the model --> linearisation
        F = self.state_transition(raw_drive_meas)

        # Model COVAR
        Q = self.predict_covariance(raw_drive_meas)

        # Drive and update the state!
        self.robot.drive(raw_drive_meas)

        # State COVAR
        self.P = F @ self.P @ F.T + Q

        '''BL: Print out the state continously here'''
        ''' Well, this state is not yet updated!'''
        # print(f"EKF state: {self.robot.state[0]} - {self.robot.state[1]} - {np.rad2deg(self.robot.state[2]) % 360}")
        # print(f"EKF state: {self.robot.state[0]} - {self.robot.state[1]} - {self.robot.state[2]}")



    '''
    Input:
        - measurements: return from ARUCO_DET()
            * lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
            * list of Marker() objects, which has
                * position and tag - input by ARUCO_DET()
                * covar: preset at 0.1*np.eye(2)
    Output:
        - self update
            * self.robot.state
            * self.P
    Description:
        - Follow the lecture slides
        
        - Added map lock by Christopher
        - Added logging by Christopher
    '''
    def update(self, measurements):
        if not measurements:
            return

        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements (landmarks pos) 
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        # Meas COVAR
        R = np.zeros((2*len(measurements),2*len(measurements)))
        for i in range(len(measurements)):
            # Stack x and y
            # @   x1  y1  x2  y2  x3
            # x1  s1  -   -   -   - 
            # y1  -   s1  -   -   - 
            # x2  -   -   s2  -   - 
            # y2  -   -   -   s2  - 
            # x3  -   -   -   -   s3
            R[2*i:2*i+2, 2*i:2*i+2] = measurements[i].covariance # return from ARUCO_DET() <------ *Question*
        # Is this H_w W^ H_w^T? <---------------- *Question*

        
        # Measurement
        z_hat = self.robot.measure(self.markers, idx_list)
        z_hat = z_hat.reshape((-1,1),order="F")
        # Measurement Jacobian of marker 
        H = self.robot.derivative_measure(self.markers, idx_list)  # H_w
        # H_w = self.robot.derivative_measure(self.markers, idx_list) # H_w
        # H_x = self.robot.derivative_drive(raw_drive_meas)          # H_x
        # Just a term in calculating Kalman Gain
        S = H @ self.P @ H.T + R
        # Textbook S
        # S = H_x @ P @ H_x.T + H_w @ R @ H_w.T
        # Kalman gain:
        K = self.P @ H.T @ np.linalg.inv(S)

        ############
        ## freeze the position of the landmarks!
        ## this will need to be met with real life tuning
        ## of the covariance values given to the markers

        ## create mask to be used across x and K
        mask = np.zeros_like(x, dtype=bool)
        mask[:3] = True # this represents the three elements of the 
                        # robots pose, which we do want to update
        mask = mask.squeeze()

        # print(mask)
        # print(K)
        # input("Enter to continue")
    
        '''Correct State'''
        x = self.get_state_vector()
        if self.lock_map:
            # Only update robot pose
            x[mask] = x[mask] + np.dot(K[mask], (z - z_hat))
        else:
            # Original code where x is updated based on measurement
            x = x + K @ (z - z_hat)

        ''' Update '''
        self.set_state_vector(x)
        # State COVAR
        P = (np.eye(x.shape[0]) - K @ H) @ self.P
        self.P = P + 0.01*np.eye(self.state_num)       # <------------------ *Question* 0.01 is a tuning parameter
        

        ## ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        ## logging/monitoring
        
        innovation = z - z_hat
        self.v.append(innovation)

        S_inv = np.linalg.inv(S)
        NIS = np.dot(np.dot(innovation.T, S_inv), innovation)
        self.nis.append(NIS)
        ## ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    

    def state_transition(self, raw_drive_meas):
        """Calculates the Jacobian of the state transition function"""
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(raw_drive_meas)
        return F
    
    def predict_covariance(self, raw_drive_meas):
        """Predicts covariance of the state after a drive command? """
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas)+ 0.01*np.eye(3)
        # tune for better? 1cm uncertainty for the state [x, y, theta]

        return Q

    def init_landmarks(self, aruco_np_array):
        # The input aruco_np_array is a numpy array of shape (10,2)
        
        self.markers = aruco_np_array.T
        
        taglist_num = aruco_np_array.shape[0]
        self.taglist = [i for i in range(1, taglist_num+1)]
        

    # Dont need to add landmarks now
    # def add_landmarks(self, measurements):
    #     if not measurements:
    #         return

    #     th = self.robot.state[2]
    #     robot_xy = self.robot.state[0:2,:]
    #     R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

    #     # Add new landmarks to the state
    #     for lm in measurements:
    #         if lm.tag in self.taglist:
    #             # ignore known tags
    #             continue
            
    #         lm_bff = lm.position
    #         lm_inertial = robot_xy + R_theta @ lm_bff

    #         self.taglist.append(int(lm.tag))
    #         self.markers = np.concatenate((self.markers, lm_inertial), axis=1)

    #         # Create a simple, large covariance to be fixed by the update step
    #         self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
    #         self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
    #         self.P[-2,-2] = self.init_lm_cov**2
    #         self.P[-1,-1] = self.init_lm_cov**2


    ####################################################################################
    ####################################################################################
    ##################      Other basic funcs      #####################################
    ####################################################################################
    ####################################################################################

    # This is only used in Operate, when we press r to reset SLAM function
    # MIGHT BE UNUSED for now
    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.taglist = []
        # Covariance matrix
        #self.P = np.zeros((3,3))
        self.P = np.eye(self.state_num)*1e3
        self.init_lm_cov = 1e3
        self.robot_init_state = None
        
    # Save the map to a file
    def save_map(self, fname="slam_map.txt"):
        if self.number_landmarks() > 0:
            utils = MappingUtils(self.markers, self.P[3:,3:], self.taglist)
            utils.save(fname)

    # Load the map from a file
    # MIGHT BE UNUSED for now
    def recover_from_pause(self, measurements):
        if not measurements:
            return False
        else:
            lm_new = np.zeros((2,0))
            lm_prev = np.zeros((2,0))
            tag = []
            for lm in measurements:
                if lm.tag in self.taglist:
                    lm_new = np.concatenate((lm_new, lm.position), axis=1)
                    tag.append(int(lm.tag))
                    lm_idx = self.taglist.index(lm.tag)
                    lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
            if int(lm_new.shape[1]) > 2:
                R,t = self.umeyama(lm_new, lm_prev)
                theta = math.atan2(R[1][0], R[0][0])
                self.robot.state[:2]=t[:2]
                self.robot.state[2]=theta
                return True
            else:
                return False

    # Return the number of landmarks saved in the class
    def number_landmarks(self):
        return int(self.markers.shape[1])

    # Return the state vector - including robot pose and all landmarks positions
    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    # Set the state vector - including robot pose and all landmarks positions
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')

    ##########################################
    ##########################################
    ##########################################

    @staticmethod
    def umeyama(from_points, to_points):

    
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
    
        return R, t

    # Plotting functions
    # ------------------
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
    
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res = (320, 500), not_pause=True):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
            
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy*0
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        # print(axes_len[0]*m2pixel)
        # print(angle)
        canvas = cv2.ellipse(canvas, start_point_uv, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                '''BL change here'''
                # Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                # axes_len, angle = self.make_ellipse(Plmi)
                # canvas = cv2.ellipse(canvas, coor_, 
                #     (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                #     angle, 0, 360, (244, 69, 96), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                    (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1],
                    (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1],
                    (coor_[0]-5, coor_[1]-5))
        return surface

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image       

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle

 
