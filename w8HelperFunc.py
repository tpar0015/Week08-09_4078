# M4 - Autonomous fruit searching
# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
# import util.w8HelperFunc as w8
import navigate_algo as navi
from Prac4_Support.Obstacle import *
from Prac4_Support.math_functions import *


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from pibot import PenguinPi    # access the robot
import DatasetHandler as dh    # save/load functions
import measure as measure      # measurements
# import pygame                       # python package for GUI


########################################################################
# Path planning

'''
Input:
    - target_fruit_list, target_fruits_pos, obstacles
    - initial_robot_pos:    exclude theta for now
    - robot_step_size:      steps per waypoint
    - ccw:                  direction for "bug" to wrap around Obstacles
    - goal_tolerance:       distance to goal to consider as reached !
'''
def get_path(target_fruit_list, target_fruit_pos, obstacles, initial_robot_pos = [0,0], 
             ccw=False,
             robot_step_size=0.05, 
             goal_tolerance=0.5,
             wrap = True):

    ###########################################################################################
    '''Create a dictionary of waypoints, each key is the fruit in search_list'''
    ###########################################################################################
    waypoint = {}
    for fruit in target_fruit_list:
        waypoint[fruit] = np.zeros((1,2))
    # print(waypoint)

    step_list = []
    path_warp_goal = np.zeros((1,2))
    # prev_waypoint = 0
    
    # Generate path - list of waypoints
    for fruit, target in zip(target_fruit_list, target_fruit_pos):
        goal_pos = target

        # The code below finds the path using bug2 algorithm
        path = navi.bug2_algorithm(goal_pos, initial_robot_pos, robot_step_size, obstacles, ccw, goal_tolerance)
        
        # print(f"Debugging ---- type: {type(path)} ---- initial shape: {path.shape}")


        waypoint[fruit] = path

        # Append to the list of steps
        step = len(path)
        step_list.append(step)

        # update
        initial_robot_pos = goal_pos
        print(f"Reached target fruit {goal_pos} after {len(path)} steps\n")

    ###########################################################################################
    ''' Interconnect between each path '''
    ###########################################################################################
    if wrap:
        for fruit_idx, fruit in enumerate(target_fruit_list):
            if fruit_idx == len(target_fruit_list)-1:
                break
            ###############################################
            # Get interconnect goal pos
            interconnect_goal_pos = target_fruit_pos[fruit_idx]
            # Create circle
            '''BL - need better upgrade, use this for now :(
            ==> compute the number of vertice so that arc length ~ robot_step_size
            '''
            goal_circle_radius = goal_tolerance
            tmp = np.arcsin(robot_step_size * 0.5 / goal_circle_radius)
            angle_each_verctice = 2 * tmp
            num_vertices = int(np.round(2*np.pi / angle_each_verctice) / 2)
            interconnect_goal = Circle(c_x=interconnect_goal_pos[0], c_y=interconnect_goal_pos[1], radius=goal_circle_radius, num_vertices=num_vertices)
            ###############################################
            # Get path of current fruit
            path = waypoint[fruit]
            
            # Get next path in waypoint
            next_path = waypoint[target_fruit_list[fruit_idx+1]]
            # Clip those start point (that suppose to be inside the circle)
            while True:
                if len(next_path) == 1: 
                    break
                if navi.compute_distance_between_points(next_path[0], interconnect_goal_pos) > goal_tolerance: 
                    break
                else: 
                    next_path = np.delete(next_path, 0, axis = 0)
            ###############################################
            # Find closest point on circle to path_end_point
            wrap_start_idx, wrap_start_point = find_nearest(interconnect_goal.vertices, path[-1])
            # Find closest point on circle to next_path_start_point
            wrap_end_idx, wrap_end_point = find_nearest(interconnect_goal.vertices, next_path[0])
            # print(wrap_start_idx, wrap_end_idx)
            
            # Create wrap around goal path
            wrap_path = shortest_arc_points(interconnect_goal_pos, goal_tolerance, wrap_start_point, wrap_end_point, num_points=num_vertices)
            # wrap_path = interconnect_goal.vertices[wrap_end_idx : wrap_start_idx]
            ###############################################
            # Update
            waypoint[target_fruit_list[fruit_idx+1]] = next_path
            waypoint[fruit] = np.append(path, wrap_path, axis = 0)
        
    return waypoint, step_list


# Set up obstacles
def get_obstacles(aruco_true_pos, fruits_true_pos, target_fruits_pos, shape = "rectangle", size = 0.4):
    # Combine aruco landmark and remained fruit as obstacles
    obs_pos = np.array(aruco_true_pos)

    # Find fruits that are not in target list
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
        # 
        if (shape.lower() == "rectangle"):
            obstacles.append(Rectangle(center=obs, width=size, height=size))

        elif (shape.lower() == "circle"):
            obstacles.append(Circle(c_x=obs[0], c_y=obs[1], radius=size/2))

    # print("Obstacles len: ", len(obstacles))
    return obstacles, obs_pos


def plot_waypoint(waypoint, target_fruit_list, target_fruits_pos, obs_pos, obstacles):
    # Create a list of 5 different marker color
    marker_color = ['y-', 'c-', 'm-', 'k-', 'g-']
    # marker_color = ['yo', 'co', 'mo', 'ko', 'w', 'g' ]

    # Plot the current path
    # i = 0
    marker_color_idx = 0
    for fruit, path in waypoint.items():
        plt.plot(path[:,0], path[:,1], marker_color[marker_color_idx], alpha=0.5)
        marker_color_idx += 1

    # Show legend with marker_color list as fruit name
    plt.legend(target_fruit_list, loc='upper left')

    # ###################################################################################
    # Plot all the obstacles
    for obs in obs_pos:
        plt.plot(obs[0], obs[1], 'bx')  
    for obstacle_outline in obstacles:
        plt.plot(obstacle_outline.vertices[:,0], obstacle_outline.vertices[:,1], 'b-', linewidth=0.5)
    # Plot all the target fruit
    for target in target_fruits_pos:
        plt.plot(target[0], target[1], 'bo')

    plt.title("Waypoint path")
    plt.axis('equal')
    plt.show(block=False)

########################################################################3
def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

########################################################################3
def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    print(fname)
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list(sname):
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open(sname, 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    fruit_coor = []

    print("Search order:")
    print(search_list)
    print(fruit_list)
    n_fruit = 1
    print(search_list)
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                ''' #BL: added return coordinations of fruits'''
                x_pos = np.round(fruit_true_pos[i][0], 2)
                y_pos = np.round(fruit_true_pos[i][1], 2)
                fruit_coor.append([x_pos, y_pos])
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                    fruit,
                                                    x_pos,
                                                    y_pos))
                
        n_fruit += 1

    ''' #BL: added return coordinations of fruits'''
    return fruit_coor



# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
# def drive_to_point(waypoint, robot_pose):
#     # imports camera / wheel calibration parameters 
#     fileS = "calibration/param/scale.txt"
#     scale = np.loadtxt(fileS, delimiter=',')
#     fileB = "calibration/param/baseline.txt"
#     baseline = np.loadtxt(fileB, delimiter=',')
    
#     ####################################################
#     # TODO: replace with your codes to make the robot drive to the waypoint
#     # One simple strategy is to first turn on the spot facing the waypoint,
#     # then drive straight to the way point

#     # Turn toward waypoint
#     robot_angle = np.arctan((waypoint[1]-robot_pose[1])/(waypoint[0]-robot_pose[0])) # rad
#     robot_angle = robot_pose[2] - robot_angle

#     wheel_vel = 30 # tick
    
#     # turn towards the waypoint
#     ''' Get baseline'''
    
#     dataDir = "{}calibration/param/".format(os.getcwd())
#     fileNameB = "{}baseline.txt".format(dataDir)
#     # read baseline from numpy formation to float
#     baseline = np.loadtxt(fileNameB, delimiter=',')


#     turn_time = (baseline * robot_angle) / wheel_vel
#     print("Turning for {:.2f} seconds".format(turn_time))

#     ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    
#     # after turning, drive straight to the waypoint
#     drive_time = 0.0 # replace with your calculation
#     print("Driving for {:.2f} seconds".format(drive_time))
#     ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
#     ####################################################

#     print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


# def get_robot_pose():
#     ####################################################
#     # TODO: replace with your codes to estimate the pose of the robot
#     # We STRONGLY RECOMMEND you to use your SLAM code from M2 here <----------------



#     # update the robot pose [x,y,theta]
#     robot_pose = [0.0,0.0,0.0] # replace with your calculation
#     ####################################################
#     # image_poses = {}
#     # with open(f'{script_dir}/lab_output/images.txt') as fp:
#     #     for line in fp.readlines():
#     #         pose_dict = ast.literal_eval(line)
#     #         image_poses[pose_dict['imgfname']] = pose_dict['pose']

#     # robot_pose = image_poses[image_poses.keys()[-1]]
#     ####################################################


#     return robot_pose

''' Thomas code - GUI and driving in pooling loop'''

    # def set_robot_pose(self, state):
    #     self.ekf.robot.state[0:3, 0] = state

    # # Waypoint navigation
    # # the robot automatically drives to a given [x,y] coordinate
    # # note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
    # # fully automatic navigation:
    # # try developing a path-finding algorithm that produces the waypoints automatically
    # def get_turn_params(self, waypoint):
    #     scale = self.ekf.robot.wheels_scale
    #     baseline = self.ekf.robot.wheels_width
    #     # Get pose
    #     robot_pose = self.get_robot_pose()
    
    #     angle_to_waypoint = np.arctan2((waypoint[1]-robot_pose[1]), (waypoint[0]-robot_pose[0]))
    #     robot_angle = -robot_pose[2] + angle_to_waypoint

    #     print(f"-- DEBUG -- Angle to turn: {np.rad2deg(robot_angle)}")

    #     turn_time = baseline/2 * robot_angle / (scale * self.turn_vel)

    #     # if turn_time != 0:
    #     # Account for negative angle
    #     if turn_time < 0:
    #         turn_time *=-1
    #         turn_vel = self.turn_vel * -1
    #     else:
    #         turn_vel = self.turn_vel

    #     return turn_vel, turn_time


    # def get_dist_params(self, waypoint):
    #     scale = self.ekf.robot.wheels_scale
    #     baseline = self.ekf.robot.wheels_width

    #     # Get pose
    #     robot_pose = self.get_robot_pose()

    #     # after turning, drive straight to the waypoint
    #     robot_dist = ((waypoint[1]-robot_pose[1])**2 + (waypoint[0]-robot_pose[0])**2)**(1/2)

    #     print(f"-- DEBUG -- Dist to drive: {robot_dist * 100} cm")

    #     drive_time = robot_dist / (scale * self.wheel_vel)

    #     return drive_time


    # def drive_control(self, turn_vel, wheel_vel, vel_time, dt, turn=False):
    #     time_series = np.arange(0, vel_time, dt)
    #     #time_gui_update = []
    #     average_gui_update_time = 0.055
    #     time_series = np.arange(0, vel_time + average_gui_update_time*len(time_series), dt)

    #     # Polling loop
    #     for t in time_series:
    #         ''' UPdate pose manually first'''
    #         cur_pose = self.get_robot_pose()
    #         if turn == True:
    #             cur_pose[-1] += dt*turn_vel
    #         else:
    #             cur_pose[0] += dt*wheel_vel*np.cos(cur_pose[-1])
    #             cur_pose[1] += dt*wheel_vel*np.sin(cur_pose[-1])

    #         self.set_robot_pose(cur_pose)
    #         # print(end_point)
    #         self.pibot.set_velocity([0 + 1*(not turn), 1*turn], time=dt)
            
                
    #         # EKF
    #         # GUI
    #         # Time it
    #         #time_prev = time.time()
    #         # self.gui_update()
    #         #time_after = time.time()
    #         #time_gui_update.append(time_after - time_prev)

    #     #print(f"Average time for GUI update: {np.mean(time_gui_update)}")

    # def gui_update(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.MOUSEBUTTONDOWN:
    #             self.gui.add_waypoint()
    #             pygame.event.clear()
    #         elif event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
    #     self.gui.update_state(self.get_robot_pose())
    #     self.gui.draw()


    # ''' Added two arguments'''
    # def drive_to_point_pooling(self, waypoint, wheel_vel = 20):

    #     #######################################################################################
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # # Turn toward waypoint
    #     turn_vel, turn_time = self.get_turn_params(waypoint)
    #     self.pibot.turning_tick = turn_vel
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # after turning, drive straight to the waypoint
    #     drive_time = self.get_dist_params(waypoint)
    #     self.pibot.tick = self.wheel_vel
    #     # this similar to self.command['motion'] in prev M
    #     dt = 0.1

    #     # To turn
    #     self.drive_control(turn_vel, self.wheel_vel, turn_time, dt, turn=True)
    #     print(f"!@#$ inside drive_to_point, cur_pose: {self.get_robot_pose()}")
    #     # To drive
    #     self.drive_control(turn_vel, self.wheel_vel, drive_time, dt, turn=False)
    #     #######################################################################################


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    print(aruco_true_pos)

    search_list = read_search_list("M4_prac_shopping_list.txt") # change to 'M4_true_shopping_list.txt' for lv2&3
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following is only a skeleton code for semi-auto navigation
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue


        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
