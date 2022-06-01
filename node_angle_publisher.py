#!/usr/bin/env python3

import os 
import tensorflow  as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
import rospy 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from datetime import datetime
from datetime import timedelta

class Server:

    def __init__(self,time):
        self.image = None
        self.velocity = None
        self.time = time


    def img_preprocess(img):
        pts1 = np.float32([[0, 164], [319, 164], [0, 239], [319, 239]])
        pts2 = np.float32([[0, 0], [319, 0], [0, 239], [319, 239]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (319, 239))
        #img = cv2.GaussianBlur(result, (3, 3), 0)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        #img=img[:,:,::-1]
        img = cv2.resize(img, (159,119))
        img = img/255
        return (img)


    def image_callback(self, msg):
        self.image = msg

        self.execute()

    def execute(self):
      time_start = datetime.now()
      if time_start > self.time + timedelta(microseconds=40000) :
        bridge = CvBridge()
        try:
            frame2cv= bridge.imgmsg_to_cv2(self.image,"bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            img_tensor1 = image.img_to_array(frame2cv)
            pts1 = np.float32([[0, 164], [319, 164], [0, 239], [319, 239]])
            pts2 = np.float32([[0, 0], [319, 0], [0, 239], [319, 239]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img_tensor1, matrix, (319, 239))
            img = cv2.GaussianBlur(result, (3, 3), 0)
            img = cv2.resize(img, (159,119))
            img = img/255
            img_tensor = np.expand_dims(img, axis=0)
            loaded_model = tf.keras.models.load_model('/home/ubuntu20/catkin_ws/src/turtlebot3/turtlebot3_example/nodes/model_train_v20.h5')
            predictions = loaded_model.predict(img_tensor)
            result=predictions[0]
            result=result[0]
            print(result)
            move.angular.z=result
            move.linear.x=0.12
        pub.publish(move)
        self.time = datetime.now()





if __name__ == '__main__':

    server = Server(datetime.now())
    rospy.init_node('movement_node')
    sub=rospy.Subscriber('/camera/image',Image,server.image_callback)
    pub=rospy.Publisher('/cmd_vel', Twist)
    move=Twist()
    rospy.spin()




