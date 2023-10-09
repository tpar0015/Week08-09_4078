   '''
    - from inputs (starting robot's pose, end point)
    - theta at endpoint is along the LINE_TO_WAYPOINT
    '''
    def get_end_pose(self, start_pose, end_point, debug = False):
        if debug:
            print("------------------------\nStarting manual set up")
            print(f"Start pose: {start_pose}")
            print(f"End_point: {end_point}")

        x = end_point[0]
        y = end_point[1]
        # do pretty much the same calculation as in drive_to_point()
        angle_to_waypoint = np.arctan2((end_point[1]-start_pose[1]),(end_point[0]-start_pose[0])) # rad

        print(f"angle to waypoint: {np.rad2deg(angle_to_waypoint)}")

        return [x, y, angle_to_waypoint]
    

    def manual_set_robot_pose(self, start_pose, end_point, debug = False):
        manual_pose = self.get_end_pose(start_pose, end_point, debug=False)
        # update robot pose when reach end point
        self.ekf.robot.state[0] = manual_pose[0]
        self.ekf.robot.state[1] = manual_pose[1]
        # self.ekf.robot.state[2] = - start_pose[2] + angle_to_waypoint
        self.ekf.robot.state[2] = manual_pose[2]

        if debug:
            print("\nDone manual set up !!!!!!!!")
            print(f"Current robot pose: {self.get_robot_pose()}\n----------------")
            input("Enter to continue")
    
    #####################################################################################
    ## From Thomas

    def pose_difference(self, end_pose):
        """Calculate the difference between the current pose and the end pose."""
        pose = self.get_robot_pose()

        angle_robot_to_point = np.arctan2(end_pose[1] - pose[1], end_pose[0] - pose[0])
        angle_robot_to_world = pose[-1]
        
        print(f"angle_waypoint {np.rad2deg(angle_robot_to_point):.3f}", end="\t")
        print(f"angle_robot {np.rad2deg(angle_robot_to_world):.3f}", end="\t")
        angle_diff = angle_robot_to_point - angle_robot_to_world

        # angle_diff = (angle_diff * np.pi) % ( 2 * np.pi) - np.pi

        print(f"Final angle_diff: {np.rad2deg(angle_diff):.3f}")

        dist = np.linalg.norm(end_pose[0:2] - pose[0:2])
        
        return dist, angle_diff


    def control(self, end_pose):
        """Control the robot to drive to the target."""
        # dt = 0.01
        # Tune this  ###############
        angle_threshold = (np.pi/180) * 5
        distance_threshold = 0.03
        self.control_clock = time.time()
        #############################
        dist_diff, ang_diff = self.pose_difference(end_pose)
        print("---------------------------")
        
        end = False
        turn_flag = True #start off wit rotating first

        while not end:
 
            # Turn or drive
            lv, rv = self.pibot.set_velocity([0 + 1*(not turn_flag), 1*turn_flag])
            # lv, rv = self.pibot.set_velocity([0, 1])

            # Localize
            # pooling ---------------------
            self.take_pic()
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(lv, -rv, dt) # physical robot - reverse wheel
            self.control_clock = time.time()
            # Get slam pose
            self.update_slam(drive_meas, print_period = 0.5)    
            # Update GUI
            # self.update_gui()
            pose = self.get_robot_pose()
            # -----------------------------
            # # Update loop conditions
            # dist_diff, ang_diff = self.pose_difference(end_pose)
            # # Check threshold
            # if (abs(ang_diff) < angle_threshold): 
            #     turn_flag = False
            #     end = True
            #     break
            # elif (abs(dist_diff) > distance_threshold):
            #     turn_flag = False
            #     print("turn_flag False")
            # else:
            #     end = True   

        print(f"Stopped with dist diff: {dist_diff}, angle diff: {np.rad2deg(ang_diff)}")
        print("---------------------------")

        self.stop()