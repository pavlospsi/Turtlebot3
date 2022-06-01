#!/usr/bin/env python3
#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from datetime import timedelta
from csv import writer
import numpy as np


class Server:
    def __init__(self,time):
        self.image = None
        self.time = time

    def image_callback(self, msg):
        self.image = msg

        self.execute()

    def execute(self):
      
      time_start = datetime.now()
      if time_start > self.time + timedelta(microseconds=700000) :
        bridge = CvBridge()
        try:
            cv2_img=self.image
            cv2_img = bridge.imgmsg_to_cv2(cv2_img, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            now = rospy.get_rostime()
            print("Saving...")
            cv2.imwrite('./sign_photo_upd/'+str(now)+'.jpeg',cv2_img)
        self.time = datetime.now()

if __name__ == '__main__':
   rospy.init_node('listener')

   server = Server(datetime.now())

   rospy.Subscriber('/camera/image',Image, server.image_callback)


   rospy.spin()

