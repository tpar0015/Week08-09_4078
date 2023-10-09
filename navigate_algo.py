from util.Prac4_Support.Obstacle import *
# from Prac4_Support.path_animation import *
from shapely import geometry



def bug2_algorithm(goal_pos, initial_robot_pos, robot_step_size, obstacles, ccw, tolerance = 0.05, terminate_step = 150):
    # Start algorithm
    path_dir = ccw
    
    ''' Split into 2 cases of "wrapping direction" '''
    robot_path = [initial_robot_pos]
    current_robot_pos = initial_robot_pos
    robot_to_goal_line = compute_line_through_points(current_robot_pos, goal_pos)


    counter = 1
    flag = 0

    # While goal not reached
    while not has_reached_goal(current_robot_pos, goal_pos, robot_step_size, tolerance):

        # Move towards goal
        next_robot_pos = move_towards_goal(current_robot_pos, robot_to_goal_line, goal_pos, initial_robot_pos, robot_step_size)
        if is_about_to_hit_obstacle(next_robot_pos, obstacles, robot_step_size, ccw):

            # Go back to current position
            next_robot_pos = np.copy(current_robot_pos)
            # Compute distance from hit point to goal
            hit_dist_to_goal = compute_distance_between_points(current_robot_pos, goal_pos)
            while True:
                # Find nearest_obstacle and start navigating around 
                closest_obs, (closest_obs_distance, obst_segment) = find_closest_obstacle(next_robot_pos,
                                                                                        obstacles, ccw)
                # Get direction along obstacle
                direction_around_obstacle = closest_obs.compute_tangent_vector_to_polygon(next_robot_pos, 
                                                                                        obst_segment)
                # Move along obstacle
                next_robot_pos = next_robot_pos + robot_step_size * direction_around_obstacle
                    
                robot_path.append(next_robot_pos)
                
                counter += 1
                if counter > terminate_step:
                    print(f"Terminating this path using ccw {ccw}")
                    flag = 1
                    break

                # Stop getting around obstacle when:
                # * We are back to the original line
                # * We are closer to the goal than when we started getting around obstacle.
                # * Segment_not_line=True meaning the robot will only stop getting around the obstacle
                #   when it hits the segment again, not the extended segment (a line).
                stop_getting_around = go_back_to_goal(next_robot_pos, goal_pos, initial_robot_pos, robot_to_goal_line,
                                                        hit_dist_to_goal, robot_step_size, segment_not_line=False)

                if stop_getting_around:
                    break

        if flag == 1:
            break

        # Update current state and add to path
        current_robot_pos = next_robot_pos
        robot_path.append(current_robot_pos)
    
    if flag == 0:
        path = np.array(robot_path)
        return path
    
    # Do it again :( but in the other direction
    robot_path = [initial_robot_pos]
    current_robot_pos = initial_robot_pos
    robot_to_goal_line = compute_line_through_points(current_robot_pos, goal_pos)

    print(f"Try again using ccw {not ccw}")

    # While goal not reached
    while not has_reached_goal(current_robot_pos, goal_pos, robot_step_size, tolerance):

        # Move towards goal
        next_robot_pos = move_towards_goal(current_robot_pos, robot_to_goal_line, goal_pos, initial_robot_pos, robot_step_size)
        if is_about_to_hit_obstacle(next_robot_pos, obstacles, robot_step_size, ccw):

            # Go back to current position
            next_robot_pos = np.copy(current_robot_pos)
            # Compute distance from hit point to goal
            hit_dist_to_goal = compute_distance_between_points(current_robot_pos, goal_pos)
            while True:
                # Find nearest_obstacle and start navigating around 
                closest_obs, (closest_obs_distance, obst_segment) = find_closest_obstacle(next_robot_pos,
                                                                                        obstacles, ccw)
                # Get direction along obstacle
                direction_around_obstacle = closest_obs.compute_tangent_vector_to_polygon(next_robot_pos, 
                                                                                        obst_segment)
                # Move along obstacle
                next_robot_pos = next_robot_pos + robot_step_size * direction_around_obstacle
                    
                robot_path.append(next_robot_pos)

                # Stop getting around obstacle when:
                # * We are back to the original line
                # * We are closer to the goal than when we started getting around obstacle.
                # * Segment_not_line=True meaning the robot will only stop getting around the obstacle
                #   when it hits the segment again, not the extended segment (a line).
                stop_getting_around = go_back_to_goal(next_robot_pos, goal_pos, initial_robot_pos, robot_to_goal_line,
                                                        hit_dist_to_goal, robot_step_size, segment_not_line=False)

                if stop_getting_around:
                    break

        # Update current state and add to path
        current_robot_pos = next_robot_pos
        robot_path.append(current_robot_pos)
        
    path = np.array(robot_path)
    return path

####################################################################################
# Helper functions
####################################################################################

def find_closest_obstacle(position, obstacle_list, ccw):
    results = [obs.compute_distance_point_to_polygon(position, ccw) for obs in obstacle_list]
    closest_obs = np.argmin([v[0] for v in results])
    return obstacle_list[closest_obs], results[closest_obs]

def has_reached_goal(current_pos, goal, step_size, tolerance):
    if compute_distance_between_points(current_pos, goal) > step_size + tolerance:
        return False
    return True

def move_towards_goal(current_pos, goal_line, goal_pos, initial_robot_pos, step_size):
    direction_to_goal = get_direction_from_line(goal_line)
    start_goal = np.array(goal_pos) - np.array(initial_robot_pos)
    start_current = np.array(current_pos) - np.array(initial_robot_pos)
    dir = 1 if np.linalg.norm(start_goal) > (np.dot(start_current, start_goal) / np.linalg.norm(start_goal)) else -1
    next_position = current_pos + step_size * direction_to_goal * dir
    return next_position


def is_about_to_hit_obstacle(next_pos, obstacle_list, step_size, ccw):
    obs, (_, _) = find_closest_obstacle(next_pos, obstacle_list, ccw)
    point = geometry.Point(next_pos)
    polygon = geometry.Polygon(obs.vertices)
    if point.within(polygon) or point.touches(polygon):
        return True
    else:
        return False

def go_back_to_goal(next_pos, goal_pos, initial_robot_pos, start_to_goal_line, distance_to_hit_point, step_size, segment_not_line = False):
    a, b, c = start_to_goal_line
    dist_to_robot_goal_line = np.abs(a*next_pos[0] + b*next_pos[1] - c)/math.sqrt(a*a + b*b)
    new_dist_to_goal = compute_distance_between_points(next_pos, goal_pos)
    stop_following_obstacle = (dist_to_robot_goal_line <= step_size) and (new_dist_to_goal < distance_to_hit_point)

    # This is to change if you consider goal line a line or a segment
    # The behaviour will be different depending on the scenario
    if segment_not_line:
        start_goal = np.array(goal_pos) - np.array(initial_robot_pos)
        start_current = np.array(next_pos) - np.array(initial_robot_pos)
        within_segment = np.linalg.norm(start_goal) > (np.dot(start_current, start_goal) / np.linalg.norm(start_goal))
        stop_following_obstacle = stop_following_obstacle and within_segment
        
    return stop_following_obstacle