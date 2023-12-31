# for computing the wheel calibration parameters
import numpy as np
import os
import sys
sys.path.insert(0, "../util")
from pibot import PenguinPi



def calibrateWheelRadius(dist):
    # Compute the robot scale parameter using a range of wheel velocities.
    # For each wheel velocity, the robot scale parameter can be computed
    # by comparing the time and distance driven to the input wheel velocities.

    ##########################################
    # Feel free to change the range / step
    ##########################################
    wheel_velocities_range = range(20, 80, 15)
    delta_times = []
    length_calibration = dist  # meters

    for wheel_vel in wheel_velocities_range:
        print("Driving at {} ticks/s.".format(wheel_vel))
        # Repeat the test until the correct time is found.
        while True:
            delta_time = input("Input the time to drive in seconds: ")
            try:
                delta_time = float(delta_time)
            except ValueError:
                print("Time must be a number.")
                continue

            # Drive the robot at the given speed for the given time
            ppi.tick = wheel_vel
            ppi.set_velocity([1, 0], time=delta_time)

            uInput = input(f"Did the robot travel {length_calibration}m?[y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print(f"Recording that the robot drove {length_calibration}m" \
                      "in {delta_time:.2f} seconds at wheel speed {wheel_vel}.\n")
                break

    # Once finished driving, compute the scale parameter by averaging
    num = len(wheel_velocities_range)
    scale = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        # pass # TODO: replace with your code to compute the scale parameter using wheel_vel and delta_time
        scale += length_calibration / (wheel_vel * delta_time)
    
    scale = scale / num
    
    print("The scale parameter is estimated as {:.6f} m/ticks.".format(scale))

    return scale



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--dist", metavar='', type=int, default=0.5)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd())

    print('Calibrating PiBot scale...\n')
    scale = calibrateWheelRadius(args.dist)
    fileNameS = "{}scale.txt".format(dataDir)
    np.savetxt(fileNameS, np.array([scale]), delimiter=',')

    print('Finished calibration')
