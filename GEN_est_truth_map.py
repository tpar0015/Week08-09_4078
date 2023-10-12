
# This program used to create the estimated truth map
# Using aruco_pose_est_pose.txt and fruit_est_pose.txt

import numpy as np
import os
import json
import ast
import TargetPoseEst
import cv2
from YOLO.detector import Detector


'''
The aruco_pose_est_pose.txt has format
    {"aruco10_0": {"y": -0.6525, "x": -0.5625}, 
    "aruco1_0": {"y": -0.8099999999999999, "x": 1.1849999999999998}, 
    "aruco2_0": {"y": 0.3675, "x": 0.32999999999999996}}

The fruit_est_pose.txt has pretty much the same format:
    {
        "pumpkin_0": {
            "y": 0.8882736268576155,
            "x": 0.020718117689841425
        },
        "potato_0": {
            "y": 0.5126230973833841,
            "x": -0.6602488324194719
        },
        "lemon_0": {
            "y": -0.21756107515764891,
            "x": -0.7425173189095063
        }
    }

The output should have the format:
{"aruco10_0": {"y": -0.6525, "x": -0.5625}, 
"aruco1_0": {"y": -0.8099999999999999, "x": 1.1849999999999998}, 
"aruco2_0": {"y": 0.3675, "x": 0.32999999999999996}, 
"potato_1": {"y": -0.57, "x": -0.075}, 
"garlic_1": {"y": 0.49499999999999994, "x": 1.14}, 
"garlic_0": {"y": 1.095, "x": -1.275}}
'''


def generate_est_truth_map():
    # delete existing est_truth_map.txt
    if os.path.exists('est_truth_map.txt'):
        os.remove('est_truth_map.txt')


    # add the aruco_est_pose.txt (formatted with comma instead of last "}")
    with open('lab_output/aruco_est_pose.txt','r') as f0, open('est_truth_map.txt','a') as f1:      
        for line in f0:
            f1.write(line)

    # and add the fruit_est.txt (formatted without first "{")
    with open('lab_output/fruit_est_pose.txt','r') as f0, open('est_truth_map.txt','a') as f1:
        for line in f0:
            f1.write(line)
    print("\nest_truth_map.txt created\n")

# Check this
def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def generate_aruco_est_pose(aruco_dict, fname="aruco_est_pose.txt"):
    tag_list = aruco_dict.keys()
    coor = np.array(list(aruco_dict.values()))
    with open(f"lab_output/{fname}", 'w') as f:
        f.write("{")

        for tag, coor in aruco_dict.items():
            # only get tag if it in range from 1 to 10
            if (tag not in range(1, 11)):
                continue
            coor = coor.tolist()
            f.write(f"\"aruco{tag}_0\": ")
            # turn list type to float
            x = str(coor[0])[1:-1]
            y = str(coor[1])[1:-1]
            f.write(f"{{\"y\": {y}, \"x\": {x}}},\n")
        # f.write("}")

    print("lab_output/aruco_est_pose.txt created\n")

if __name__ == '__main__':
    import argparse

    # Arguments
    parser = argparse.ArgumentParser("Create estimate map with aruco and fruit")
    parser.add_argument("--slam", type=str, help="path to slam.txt", default="lab_output/slam.txt")
    parser.add_argument("--fruit", type=str, help="path to targets.txt", default="lab_output/targets.txt")
    args = parser.parse_args()
    # get current script directory (TargetPoseEst.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))     
    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/latest_model.pt'
    yolo = Detector(model_path)

    # create a dictionary of all the saved images with their corresponding robot pose
    # txt file that store robot pose for each predicted image
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
            # lower case is used to match the names format
            target_pose_dict[f'{detection[0].lower()}_{occurrence}'] = TargetPoseEst.estimate_pose(camera_matrix, detection, robot_pose) # lower case <------
            # print(f"--------\n{target_pose_dict}\n--------")
            detected_type_list.append(detection[0])

    # print(f"Done extracting: Target_pose_dict:")
    # for target_dict in target_pose_dict:
    #     print(f"{target_dict.split('_')[0]}: {target_pose_dict[target_dict]}")

    '''MERGING'''
    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    # target_est = {}
    # target_est = merge_estimations(target_pose_dict)
    
    target_est = target_pose_dict

    # print(f"Done merging: merged target_est")
    # for target_est in target_pose_dict:
    #     print(f"{target_est.split('_')[0]}: {target_pose_dict[target_est]}")

    # save target pose estimations
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)
    print('Estimations saved!')

    # save target pose estimations into fruit_pose_est.txt without the first "{"
    with open(f'{script_dir}/lab_output/fruit_est_pose.txt', 'w') as fo:
        json.dump(target_est, fo, indent=0)

    # remove the first "{"
    with open(f'{script_dir}/lab_output/fruit_est_pose.txt', 'r+') as fo:
        content = fo.read()
        fo.seek(0, 0)
        fo.write(content[1:-1])
    
    print('fruit_est_pose.txt saved!')

    us_aruco = parse_user_map(args.slam)

    generate_aruco_est_pose(us_aruco)

    generate_est_truth_map()
    
    # # print out the est_truth_map.txt
    # with open('est_truth_map.txt','r') as f:
    #     for line in f:
    #         print(line, end='')