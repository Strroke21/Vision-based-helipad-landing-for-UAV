###########DEPENDENCIES################
import time
import math
import argparse

from dronekit import connect, VehicleMode,LocationGlobalRelative
from pymavlink import mavutil
from ultralytics import YOLO

import cv2
import numpy as np
import subprocess
#from imutils.video import WebcamVideoStream
#import imutils
#######VARIABLES####################
# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#camera matrix #load the camera matrix from calibration file
camera_matrix = np.array([[6.482266709864785525e+02, 0.000000000000000000e+00, 6.283597219044611393e+02],
                          [0.000000000000000000e+00, 6.494701681927931531e+02, 3.399607798824992528e+02],
                          [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

# Load YOLO model
model = YOLO(r"C:\Users\ADMIN\Desktop\python_scripts\yolo-Weights\test03.pt")

classNames = ["helipad"]

takeoff_height = 6
velocity = 0.5


hfov = 62.2 * (math.pi / 180 ) ##Pi cam V1: 53.5 V2: 62.2
vfov = 48.8 * (math.pi / 180)    ##Pi cam V1: 41.41 V2: 48.8

width = 1280
height = 720


script_mode = 1##1 for arm and takeoff, 2 for manual LOITER to GUIDED land 
ready_to_land=0 ##1 to trigger landing

manualArm=False ##If True, arming from RC controller, If False, arming from this script.


#########FUNCTIONS#################

def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect

    if not connection_string:
        connection_string = '/dev/serial/by-id/usb-CubePilot_CubeOrange+_330024001751313437363430-if00'

    vehicle = connect(connection_string, wait_ready=True)

    return vehicle

def get_distance_meters(targetLocation,currentLocation):
    dLat=targetLocation.lat - currentLocation.lat
    dLon=targetLocation.lon - currentLocation.lon

    return math.sqrt((dLon*dLon)+(dLat*dLat))*1.113195e5

def goto(targetLocation):
    distanceToTargetLocation = get_distance_meters(targetLocation,vehicle.location.global_relative_frame)

    vehicle.simple_goto(targetLocation)

    while vehicle.mode.name=="GUIDED":
        currentDistance = get_distance_meters(targetLocation,vehicle.location.global_relative_frame)
        if currentDistance<distanceToTargetLocation*.02:
            print("Reached target waypoint.")
            time.sleep(2)
            break
        time.sleep(1)
    return None


def arm_and_takeoff(targetHeight):
    while vehicle.is_armable != True:
        print("Waiting for vehicle to become armable.")
        time.sleep(1)
    print("Vehicle is now armable")
    
    vehicle.mode = VehicleMode("GUIDED")
            
    while vehicle.mode != 'GUIDED':
        print("Waiting for drone to enter GUIDED flight mode")
        time.sleep(1)
    print("Vehicle now in GUIDED MODE. Have fun!!")

    if manualArm == False:
        vehicle.armed = True
        while vehicle.armed == False:
            print("Waiting for vehicle to be armed")
            time.sleep(1)
    else:
        if vehicle.armed == False:
            print("Exiting script. manualArm set to True but vehicle not armed.")
            print("Set manualArm to True if desiring script to arm the drone.")
            return None

    print("Propellers are spinning...")
    vehicle.simple_takeoff(targetHeight)

    while True:
        print("Current Altitude: %d" % vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= 0.95 * targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!!")

    return None




def send_local_ned_velocity(vx, vy, vz):
	msg = vehicle.message_factory.set_position_target_local_ned_encode(
		0,
		0, 0,
		mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
		0b0000111111000111,
		0, 0, 0,
		vx, vy, vz,
		0, 0, 0,
		0, 0)
	vehicle.send_mavlink(msg)
	vehicle.flush()
  

#################### Landing Target Function ##############  
  
def send_land_message(x,y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x,
        y,
        0,
        0,
        0,)
    vehicle.send_mavlink(msg)
    vehicle.flush()



def lander():

    if vehicle.mode!='LAND':
        vehicle.mode=VehicleMode("LAND")
        while vehicle.mode!='LAND':
            print('WAITING FOR DRONE TO ENTER LAND MODE')
            time.sleep(1)

    success, img = cap.read()
    altitude = vehicle.rangefinder.distance
    results = model(img, stream=True)

    # Get image dimensions
    #height, width, _ = img.shape

    # Iterate through detected objects
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

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

                x_ang = (x_avg - width * 0.5) * (hfov / width)
                y_ang = (y_avg - height * 0.5) * (vfov / height)
                

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
    
     

####################### MAIN DRONE PARAMETERS ###########################

########### main vehicle parameters #####
vehicle = connectMyCopter()
    ##SETUP PARAMETERS TO ENABLE PRECISION LANDING
vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1 ##1 for companion computer
vehicle.parameters['PLND_EST_TYPE'] = 0 # 0 for raw sensor, 1 for kalman filter pos estimation
vehicle.parameters['LAND_SPEED'] = 30 ##Descent speed of 30cm/s


#########parameters for fake rangefinder ##########
# vehicle.parameters['RNGFND2_TYPE'] = 10
# vehicle.parameters['RNGFND2_MIN_CM'] = 20 
# vehicle.parameters['RNGFND2_MAX_CM'] = 1000
# vehicle.parameters['RNGFND2_GNDCLEAR'] = 10
####################################################


############### first 3D fix location as home location (Static Home Location) ######### 
home_lat= vehicle.location.global_relative_frame.lat
home_lon= vehicle.location.global_relative_frame.lon
wp_home = LocationGlobalRelative(home_lat,home_lon,takeoff_height)
###################################################


if script_mode ==1:
    arm_and_takeoff(takeoff_height)
    print(str(time.time()))
    #send_local_ned_velocity(velocity,velocity,0) ##Offset drone from target
    time.sleep(1)
    ready_to_land=1


elif script_mode ==2:
    
    while True:
        
        ########### home location coordinates (Dynamic) ########
        #home_lat= vehicle.home_loaction.lat
        #home_lon= vehicle.home_location.lon
        #wp_home = LocationGlobalRelative(home_lat,home_lon,takeoff_height)
        ##############################################
        
        ########### current location from drone #######
        lat_current =vehicle.location.global_relative_frame.lat
        lon_current=vehicle.location.global_relative_frame.lon
        current_altitude = vehicle.location.global_relative_frame.alt
       
        ######### distance_to_home calculation ########
        wp_current = LocationGlobalRelative(lat_current,lon_current, current_altitude)
        distance_to_home = get_distance_meters(wp_current,wp_home)
        altitude = vehicle.rangefinder.distance
          
        if (vehicle.mode=='RTL' and distance_to_home<= 3 and altitude<=8) or (vehicle.mode=='LAND'): 
            
            print("Landing Point Acquired...")
            ready_to_land=1
            break
        
        time.sleep(1)
        print("Distance to Home:"+str(distance_to_home)+ " Altitude:" +str(altitude)+" Waiting to acquire Landing Point...")
        
              
       
if ready_to_land==1:
    
    while vehicle.armed==True:
        lander()
        time.sleep(5)
    
    print("------------------")
    print("Precision landing completed...")
    #subprocess.call['sudo','reboot']
    



           




