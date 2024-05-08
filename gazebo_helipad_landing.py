#!/usr/bin/env python

########IMORTS#########

import rospy
from sensor_msgs.msg import Image
import cv2
import cv2.aruco as aruco
import sys
import time
import math
import numpy as np
import ros_numpy as rnp
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from array import array
from ultralytics import YOLO

#######VARIABLES########

vehicle = connect('tcp:127.0.0.1:5763',wait_ready=True)
vehicle.parameters['PLND_ENABLED']=1
vehicle.parameters['PLND_TYPE']=1
vehicle.parameters['PLND_EST_TYPE']=0
vehicle.parameters['LAND_SPEED']=30 ##cms/s

velocity=-.5 #m/s
takeoff_height=4 #m
########################
newimg_pub = rospy.Publisher('/camera/color/image_new', Image, queue_size=10)

horizontal_res = 640
vertical_res = 480

hfov = 62.2 * (math.pi / 180) ##62.2 for picam V2, 53.5 for V1
vfov = 48.8 * (math.pi / 180) ##48.8 for V2, 41.41 for V1


model = YOLO(r"C:\Users\ADMIN\Desktop\python_scripts\yolo-Weights\best.pt")
classNames = ["helipad"]
#############CAMERA INTRINSICS#######

dist_coeff = [0.0, 0.0, 0.0, 0.0, 0.0]
camera_matrix = [[530.8269276712998, 0.0, 320.5],[0.0, 530.8269276712998, 240.5],[0.0, 0.0, 1.0]]
np_camera_matrix = np.array(camera_matrix)
np_dist_coeff = np.array(dist_coeff)

#####
time_last=0
time_to_wait = .1 ##100 ms
################FUNCTIONS###############
def arm_and_takeoff(targetHeight):
    while vehicle.is_armable !=True:
        print('Waiting for vehicle to become armable')
        time.sleep(1)
    print('Vehicle is now armable')

    vehicle.mode = VehicleMode('GUIDED')

    while vehicle.mode !='GUIDED':
        print('Waiting for drone to enter GUIDED flight mode')
        time.sleep(1)
    print('Vehicle now in GUIDED mode. Have Fun!')

    vehicle.armed = True
    while vehicle.armed ==False:
        print('Waiting for vehicle to become armed.')
        time.sleep(1)
    print('Look out! Virtual props are spinning!')

    vehicle.simple_takeoff(targetHeight)

    while True:
        print('Current Altitude: %d'%vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >=.95*targetHeight:
            break
        time.sleep(1)
    print('Target altitude reached!')

    return None

##Send velocity command to drone
def send_local_ned_velocity(vx,vy,vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0,
        0,
        0,
        vx,
        vy,
        vz,
        0,0,0,0,0)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def send_land_message(x,y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0,0,0
        )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def msg_receiver(message):

    if time.time() - time_last > time_to_wait:
        np_data = rnp.numpify(message) ##Deserialize image data into array
        img = cv2.cvtColor(np_data, cv2.COLOR_BGR2GRAY)

        altitude = vehicle.rangefinder.distance
        results = model(img, stream=True)

        # Get image dimensions
        height, width, _ = img.shape

        # Iterate through detected objects
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Class name
                cls = int(box.cls[0])

                if (classNames[cls] == 'helipad') and (altitude>=2):
                    # Calculate the center pixel of the object
                    object_center_x = (x1 + x2) / 2
                    object_center_y = (y1 + y2) / 2
                    #image_center_x = width / 2
                    #image_center_y = height / 2

                    object_center_px = np.array([[object_center_x], [object_center_y], [1]])
                    object_center_normalized = np.linalg.inv(camera_matrix) @ object_center_px
                    object_center_normalized /= object_center_normalized[2]  # Normalize by the homogeneous coordinate
                    object_position_camera_space = object_center_normalized * altitude


                    #distance in pixel
                    #x_px = object_center_x - width / 2
                    #y_px = object_center_y - height / 2

                    #distance in meters
                    #x_m = (x_px / width) * (2 * altitude * math.tan(math.radians(hfov / 2)))
                    x_m = object_position_camera_space[0]
                    #y_m = (y_px / height) * (2 * altitude * math.tan(math.radians(vfov / 2)))
                    y_m = object_position_camera_space[1]


                    x_sum = x1 + x2
                    y_sum = y1 + y2

                    x_avg = x_sum * 0.5
                    y_avg = y_sum * 0.5

                    x_ang = (x_avg - width*0.5) * (hfov / horizontal_res)
                    y_ang = (y_avg - height*0.5) * (vfov / vertical_res)
                    

                    if vehicle.mode!='LAND':
                        vehicle.mode = VehicleMode('LAND')
                    
                        while vehicle.mode!='LAND':
                            time.sleep(1)
                        print("------------------------")
                        print("Vehicle now in LAND mode")
                        print("------------------------")
                        send_land_message(x_ang,y_ang)
                        #send_distance_message(z_int) #fake lidar message
                    else:
                        send_land_message(x_ang,y_ang)
                        #send_distance_message(z_int) #fake lidar message
                        pass

                    print(f"x angle: {x_ang}, y angle: {y_ang}")
                    print(f"Distance X: {x_m} Distance Y: {y_m}")

    else:

        print("Target not Found...")


def subscriber():
    rospy.init_node('drone_node',anonymous=False)
    sub = rospy.Subscriber('/camera/color/image_raw', Image, msg_receiver)
    rospy.spin()


if __name__=='__main__':
    try:
        arm_and_takeoff(takeoff_height)
        time.sleep(1)
        send_local_ned_velocity(0,velocity,0)
        time.sleep(10)
        subscriber()
        time.sleep(5)
    except rospy.ROSInterruptException:
        pass
