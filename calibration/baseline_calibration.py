# for computing the wheel calibration parameters
import numpy as np
import os
import sys
sys.path.insert(0, "../util")
from pibot import PenguinPi

def calibrateBaseline(angle):
    # Compute the robot basline parameter using a range of wheel velocities.
    # For each wheel velocity, the robot baseline parameter can be computed by
    # comparing the time elapsed and rotation completed to the input wheel
    # velocities to find out the distance between the wheels.

    path = os.getcwd() + "/"
    fileS = "{}param/scale.txt".format(path)
    scale = np.loadtxt(fileS, delimiter=',')

    ##########################################
    # Feel free to change the range / step
    ##########################################
    wheel_velocities_range = range(10, 50, 10)
    delta_times = []
    angle_caliration_deg = angle # here!
    angle_caliration_rad = np.deg2rad(angle_caliration_deg)

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

            # Spin the robot at the given speed for the given time
            ppi.turning_tick=wheel_vel
            ppi.tick=20
            ppi.set_velocity([0, 1], time = delta_time)

            uInput = input(f"Did the robot spin {angle_caliration_deg} deg?[y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print(f"Recording that the robot spun {angle_caliration_deg} deg in {delta_time:.2f} seconds at wheel speed {wheel_vel}.\n")
                break

    # Once finished driving, compute the basline parameter by averaging
    num = len(wheel_velocities_range)
    baseline = 0
    for delta_time, wheel_vel in zip(delta_times, wheel_velocities_range):
        # pass # TODO: replace with your code to compute the baseline parameter using scale, wheel_vel, and delta_time
        # baseline += ( delta_time * wheel_vel * scale ) / np.pi
        baseline += 2 * ( delta_time * wheel_vel * scale ) / (angle_caliration_rad)
    
    baseline = baseline / num
    print("The baseline parameter is estimated as {:.6f} m.".format(baseline))

    return baseline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--angle", metavar='', type=int, default=180)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd())

    print('Calibrating PiBot baseline...\n')
    baseline = calibrateBaseline(args.angle)
    fileNameB = "{}baseline.txt".format(dataDir)
    np.savetxt(fileNameB, np.array([baseline]), delimiter=',')

    print('Finished calibration')
