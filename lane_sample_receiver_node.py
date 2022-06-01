#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from datetime import timedelta
from csv import writer


class Server:
    def __init__(self,time):
        self.image = None
        self.velocity = None
        self.time = time

    def image_callback(self, msg):
        self.image = msg
        self.execution()

    def velocity_callback(self, msg):
        self.velocity = msg.linear.x
        self.angular=msg.angular.z

        #Compute stuff.
        self.execution()

    def execution(self):
      
      time_start = datetime.now()
      if time_start > self.time + timedelta(microseconds=500000) :
         bridge = CvBridge()
         try:
            cv2_image = bridge.imgmsg_to_cv2(self.image, "bgr8")
            pts1 = np.float32([[0, 164], [319, 164], [30, 239], [289, 239]])
            pts2 = np.float32([[0, 0], [319, 0], [0, 239], [319, 239]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            cv2_img = cv2.warpPerspective(cv2_image, matrix, (319, 239))
         except CvBridgeError as e:
            print(e)
         else:
            turn=float("{0:.3f}".format(self.angular))
            turn_s=str(turn)
            now = rospy.get_rostime()
            print("Saving...")
            cv2.imwrite(''+str(now)+''+turn_s+'.jpeg', cv2_img)
            datalist=['pathr'+str(now)+''+turn_s+'.jpeg',str(self.velocity),turn_s]
            print(datalist,"epitixis prosthiki.")
            with open('DATA.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(datalist)
                f_object.close()
         self.time = datetime.now()

if __name__ == '__main__':
    rospy.sleep(5.)
    rospy.init_node('listener')

    server = Server(datetime.now())

    rospy.Subscriber('/camera/image',Image, server.image_callback)
    rospy.Subscriber('/cmd_vel', Twist, server.velocity_callback)


    rospy.spin()
