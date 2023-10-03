# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector


# list of target fruits and vegs types
# Make sure the names are the same as the ones used in your YOLO model
# TARGET_TYPES = ['orange', 'lemon', 'lime', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']


def estimate_pose(camera_matrix, obj_info, robot_pose):
    """
    function:
        estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient (should be 0 for PenguinPi)
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/images.txt', [x,y,theta])
    output:
        target_pose: dict, prediction of target pose
    """
    # read in camera matrix (from camera calibration results)
    focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######################################################################
    # first letter CAP!
    ######################################################################
    target_dimensions_dict = {'Orange': [0.075,0.075,0.073], 'Lemon': [0.08,0.05,0.05], 
                              'Lime': [0.08,0.05,0.05], 'Tomato': [0.07,0.07,0.065], 
                              'Capsicum': [0.095,0.085,0.085], 'Potato': [0.11,0.06,0.062], 
                              'Pumpkin': [0.07,0.085,0.075], 'Garlic': [0.08,0.065,0.075]}
    
    # pumkin include stem
    # capsicum include stem
    # lemon ~ lime  --> but hard to get precise height
    # garlic height --> TILTED!

    '''FIND HEIGHT OF OBJECT IN IMAGE'''
    # estimate target pose using bounding box and robot pose
    target_class = obj_info[0]     # get predicted target label of the box
    target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label
    # compute pose of the target based on bounding box info, true object height, and robot's pose
    pixel_height = target_box[3]
    pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length  # estimated distance between the object and the robot based on height
    # print(distance)

    '''FIND LOCATION OF OBJECT IN IMAGE - RELATIVE TO ROBOT'''
    image_width = 320 # change this if your training image is in a different size (check details of pred_0.png taken by your robot)
    x_shift = image_width/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
    theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
    ang = theta + robot_pose[2]     # angle of object in the world frame
    # relative object location
    distance_obj = distance/np.cos(theta) # relative distance between robot and object
    x_relative = distance_obj * np.cos(theta) # relative x pose
    y_relative = distance_obj * np.sin(theta) # relative y pose
    relative_pose = {'x': x_relative, 'y': y_relative}
    # print(f"theta: {np.rad2deg(theta)} --> relative_pose: {relative_pose}") # --> this is alright!!!!!!!!!!

    '''FIND LOCATION OF OBJECT IN WORLD - RELATIVE TO WORLD'''
    # location of object in the world frame using rotation matrix
    delta_x_world = x_relative * np.cos(robot_pose[2]) - y_relative * np.sin(robot_pose[2])
    delta_y_world = x_relative * np.sin(robot_pose[2]) + y_relative * np.cos(robot_pose[2])
    # add robot pose with delta target pose
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                   'x': (robot_pose[0]+delta_x_world)[0]}
    #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    #print(f'target_pose: {target_pose}')

    return target_pose



def merge_estimations(target_pose_dict):
    """
    function:
        merge estimations of the same target
    input:
        target_pose_dict: dict, generated by estimate_pose
    output:
        target_est: dict, target pose estimations after merging
    """
    target_est = {}
    ###############################################################################
    # TODO: 1. Create dict to first, store ALL the detected coor of each fruit type
    ###############################################################################
    fruit_est_dict = {'orange':[],'lemon':[],'lime':[],'tomato':[],'capsicum':[],'potato':[],'pumpkin':[],'garlic':[]}

    # Combine the estimations from multiple detector outputs
    for name, coor in target_pose_dict.items():

        # Read from the returned target_pose_dict (format: fruit_occurrence)
        fruit = name.split('_')[0]
        # Check the current dict
        if fruit in fruit_est_dict:
            xval = target_pose_dict[name]['x']
            yval = target_pose_dict[name]['y']
            fruit_est_dict[fruit].append(np.array([xval, yval]))

    ######################################################################
    # TODO 2: Go thru and merge each fruit type
    ######################################################################
    for fruit, est in fruit_est_dict.items():
        if len(est) >= 2:
            merged_estimation = average_coordinates(est)

            # Append the merged coor
            for i in range(len(merged_estimation)):
                target_est[f'{fruit}_{i}'] = {'y': merged_estimation[i][1], 'x': merged_estimation[i][0]}

    return target_est


'''
Input:
    - list of coordinates of all detected fruit of a same type
    - distance threshold - if under this threshold, merge the fruit
Output:
    - list containing "cluster" - corresponding to each occurence of that fruit type
'''
def average_coordinates(coor, threshold = 0.15):
    merged_coor = {
        "occur1": [coor[0]],
        "occur2": [],
        "occur3": [],
        "occur4": [],
        "occur5": [],
    }

    # for tmp in coor: print(tmp)
    # input("Enter pls")
    
    for idx, coor in enumerate(coor[1:]):
        # Get coordination
        y = coor[0]
        x = coor[1]

        # Append coor into correct cluster if within threshold
        # Iter through all the clusters and break once append the coor succesfully
        for key, cluster in merged_coor.items():
            if len(cluster) == 0:
                cluster.append([y,x])
                break

            # Get the last element of the cluster
            prev_y = cluster[-1][0]
            prev_x = cluster[-1][1]
            
            dist = np.sqrt((y - prev_y)**2 + (x - prev_x)**2)

            if dist < threshold:
                cluster.append([y, x])
                break

    # print(merged_coor); input("Enter pls")
    
    average_merged_coor = []
    # Average x and y in each cluster
    for key, cluster in merged_coor.items():
        if len(cluster) == 0: continue
        ysum = 0
        xsum = 0
        for coor in cluster:
            ysum += coor[0]
            xsum += coor[1]
        yavg = ysum/len(cluster)
        xavg = xsum/len(cluster)
        average_merged_coor.append([yavg, xavg])

    return np.array(average_merged_coor)


# main loop
if __name__ == "__main__":
    # get current script directory (TargetPoseEst.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))     

    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    '''Replace model here'''
    # model_path = f'{script_dir}/YOLO/model/yolov8_model.pt'
    # model_path = f'{script_dir}/YOLO/model/best_4_Sep.pt'
    model_path = f'{script_dir}/YOLO/model/latest_model.pt'
    yolo = Detector(model_path)

    # create a dictionary of all the saved images with their corresponding robot pose
    ''' #BL - from txt file that store robot pose and pred img name'''
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each image
    target_pose_dict = {}
    detected_type_list = []
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img = yolo.detect_single_image(input_image)
        # cv2.imshow('bbox', bbox_img)
        # cv2.waitKey(0)
        robot_pose = image_poses[image_path]

        for detection in bounding_boxes:
            # count the occurrence of each target type
            occurrence = detected_type_list.count(detection[0])
            '''LOWER CASE HERE'''
            target_pose_dict[f'{detection[0].lower()}_{occurrence}'] = estimate_pose(camera_matrix, detection, robot_pose)
            # print(f"--------\n{target_pose_dict}\n--------")
            detected_type_list.append(detection[0])

    print(f"Done extracting: Target_pose_dict:")
    for target_dict in target_pose_dict:
        print(f"{target_dict.split('_')[0]}: {target_pose_dict[target_dict]}")
    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = {}
    target_est = merge_estimations(target_pose_dict)

    print(f"Done merging: merged target_est")
    for target_est in target_pose_dict:
        print(f"{target_est.split('_')[0]}: {target_pose_dict[target_est]}")

    # save target pose estimations
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)

    print('Estimations saved!')