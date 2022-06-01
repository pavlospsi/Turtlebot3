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


def image_callback(frame):
    bridge = CvBridge()
    try: 
        cv2_img = bridge.imgmsg_to_cv2(frame, "bgr8")
        pts1 = np.float32([[0, 164], [319, 164], [0, 239], [319, 239]])
        pts2 = np.float32([[0, 0], [319, 0], [0, 239], [319, 239]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        cv2_final = cv2.warpPerspective(cv2_img, matrix, (319, 239))
    except CvBridgeError as e:
        print(e)
    else:
        cv2.imshow('lane view',cv2_final)
        cv2.waitKey(1) 
    return

def velocity_callback(value):
    steering= value.angular.z
    rospy.loginfo(steering)


if __name__ == '__main__':
   rospy.init_node('listener_2')

   rospy.Subscriber('/camera/image',Image,image_callback)
   rospy.Subscriber('/cmd_vel', Twist, velocity_callback)


   rospy.spin()

