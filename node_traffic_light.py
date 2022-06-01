#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from datetime import timedelta
from csv import writer
import numpy as np
from std_msgs.msg import String
np.set_printoptions(threshold=np.inf)


def image_callback(frame):
    bridge = CvBridge()
    mask=0
    r2 = rospy.Rate(0.18)
    try: 
        cv2_img = bridge.imgmsg_to_cv2(frame, "bgr8")
        gray = cv2.cvtColor(cv2_img , cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 0, maxRadius = 20)
        apofasi='no_traffic_sign'
        if detected_circles is not None:

            #print(detected_circles[1])
            #(b, g, r) = cv2_img[detected_circles[1],detected_circles[2]]
            #print("Pixel at (50, 50) - Red: {}, Green: {}, Blue: {}".format(r,g,b))
            detected_circles = np.uint16(np.around(detected_circles))
            apofasi='prasino'
            for pt in detected_circles[0, :]:
                
                a, b, aktina = pt[0], pt[1], pt[2]
                print(aktina)
                (b, g, r) = cv2_img[b, a]
                if r>250 and g>250 and b<12 and aktina <16 and aktina> 6:
                    apofasi='kokkino'
                    pub.publish(apofasi)
                    print ("kokkino fanari")
                    pub.publish(apofasi)
                    r2.sleep()
                #cv2.circle(cv2_img,  (a, b), r, (0, 255, 0), 2)
                #cv2.circle(cv2_img , (a, b), 1, (0, 0, 255), 3)
                #print(a,b)
                (b, g, r) = cv2_img[b, a]
        pub.publish(apofasi)
        

    except CvBridgeError as e:
        print(e)
    else:
        #cv2.imshow("Detected Circle", mask )
        cv2.imshow('camera view',cv2_img)
        cv2.waitKey(1) 
    return

def velocity_callback(value):
    steering= value.angular.z
    rospy.loginfo(steering)


if __name__ == '__main__':
   rospy.init_node('listener_1')

   rospy.Subscriber('/camera/image',Image,image_callback)
   rospy.Subscriber('/cmd_vel', Twist, velocity_callback)
   pub=rospy.Publisher('/fanari',String, queue_size=10 )


   rospy.spin()

