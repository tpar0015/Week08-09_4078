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


def test_opencv_circle():
        
    # Reading an image in default mode
    Img = np.zeros((512, 512, 3), np.uint8)
        
    # Window name in which image is displayed
    window_name = 'Image'
        
    # Center coordinates
    center_coordinates = (220, 150)
    
    # Radius of circle
    radius = 100
        
    # Red color in BGR
    color = (255, 133, 233)
        
    # Line thickness of -1 px
    thickness = -1
        
    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    image = cv2.circle(Img, center_coordinates, radius, color, thickness)
        
    # Displaying the image
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_plot_icons():
    paths = [
        'lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg'
        ]
        
    x = [0,1,2,3,4]
    y = [0,1,2,3,4]

    fig, ax = plt.subplots()
    ax.scatter(x, y) 

    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(w8.getImage('lemon_20x20.png'), (x0, y0), frameon=False)
        ax.add_artist(ab)

    plt.show()


if __name__ == '__main__':
    # # define circle centre as origin
    # centre = np.array([5,0])
    # circle1 = Circle(centre[0], centre[1], 3)
    # print(circle1.vertices)

    # # # Plot the origin in red and vertices in blue
    # # plt.scatter(centre[0], centre[1], color='r')
    # # plt.scatter(circle1.vertices[:,0], circle1.vertices[:,1], color='b')

    # # # Both axes same scale
    # # plt.axis('equal')
    # # plt.show()

    # # Init
    # initial_robot_pos = np.array([0, 0])
    # robot_step_size = 0.1
    # obstacles = [circle1]
    # ccw = True

    # # Check path
    # goal_pos = np.array([20, 0])
    
    # path = navi.bug2_algorithm(goal_pos, initial_robot_pos, robot_step_size, obstacles, ccw, tolerance = 0.02)

    # # Plot path
    # plt.scatter(path[:,0], path[:,1], color='b')
    # plt.scatter(centre[0], centre[1], color='r')
    # plt.scatter(goal_pos[0], goal_pos[1], color='g')
    # plt.scatter(initial_robot_pos[0], initial_robot_pos[1], color='y')
    # plt.axis('equal')
    # plt.show()
    
    x = np.linspace(0, 10, 500)
    y = np.sin(x**2)+np.cos(x)
    
    fig, ax = plt.subplots()
    
    ax.plot(x, y, label ='Line 1')
    
    ax.plot(x, y - 0.6, label ='Line 2')
    
    ax.legend()
    
    fig.suptitle("""matplotlib.figure.Figure.show()
    function Example\n\n""", fontweight ="bold") 
    
    fig.show() 
    ##################
    # dict1 = {
    #     "banana": [1,2],
    #     "apple": [2,3],
    #     "lemon": [3,4]
    # }

    # for fruit, coor in dict1.items():
    #     print(fruit)
    #     print(coor)

    if start:
        i = 0
        operate = Operate(args)
        operate.stop()
        waypoint = [0.0, 0.0]
        operate.gui.add_manual_waypoint(waypoint)
        print("\n\n~~~~~~~~~~~~~\nStarting\n~~~~~~~~~~~~~\n\n")
        input("Enter to start")

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

            '''STOP'''
            if i == len(waypoints) - 1:
                start = False