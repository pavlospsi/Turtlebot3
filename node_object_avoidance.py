#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist 
from math import atan2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from datetime import datetime
from datetime import timedelta

x = 0.0
y = 0.0
theta = 0.0
right=0.0
left=0.0
Fright=0.0
Fleft=0.0
straight = 0.0
nodestate='lane detection...'
def LaserValues(msg):
    #print (len(msg.ranges))
    global right
    global left
    global straight
    global Fleft
    global Fright
    #print ("test", len(msg.ranges),msg.ranges[0], msg.ranges[359])
    straight=min(min(msg.ranges[0:10]),min(msg.ranges[349:359]))
    #right=min(msg.ranges[269:304])
    Fright = min(msg.ranges[330:349])
    #left=min(msg.ranges[53:89])
    Fleft = min(msg.ranges[10:28])

def newOdom(msg):
    global x
    global y
    global theta

    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    #print(x,y)

    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta) = euler_from_quaternion ([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
def apofasi(msg):
    global nodestate 
    nodestate=msg.data    

rospy.init_node("tunnel_node")
sub = rospy.Subscriber("/odom", Odometry, newOdom)
sub = rospy.Subscriber("/scan",LaserScan,LaserValues)
sub1 = rospy.Subscriber("/chater",String,apofasi)
pub = rospy.Publisher("/cmd_vel",Twist, queue_size=1)
pub2 = rospy.Publisher('/tunnel', String, queue_size=10)

speed = Twist()
r = rospy.Rate(5)
goal = Point()
goal.x = 0.13
goal.y = -1.7

while not rospy.is_shutdown():
    inc_x = goal.x-x
    inc_y = goal.y-y
    #print(x,y)
    angle_to_goal = atan2 (inc_y, inc_x)
    linear_speed=0.16
    angular_speed=0.15
    threshold_dist=0.3


    print(nodestate)

    tunnel_state='ektos_tunnel'
    if nodestate=='tunnel_process':
        if abs(inc_x)>0.16 or abs(inc_y)>0.16:
            #print(inc_y, inc_x)
            if straight > threshold_dist and Fleft > threshold_dist and Fright > threshold_dist:
                state_description = 'case 1 - kanena empodio'
                speed.linear.x = linear_speed
                speed.angular.z = 0
                '''if abs(angle_to_goal - theta) > 0.1 and left<right and left>0.3:
                    speed.linear.x = 0.0
                    speed.angular.z=0.2
                    print("vriskw gwnia eksodou")
                if abs(angle_to_goal - theta) > 0.1  and left>right and right>0.3:
                    print("vriskw gwnia eksodou")
                    speed.linear.x = 0.0
                    speed.angular.z=-0.2
                else:
                    speed.linear.x = 0.15
                    speed.angular.z = 0.0'''
            elif straight < threshold_dist and Fleft > threshold_dist and Fright > threshold_dist:
                state_description = 'case 2 - Empodio eutheia mprosta'
                speed.linear.x = 0.02
                speed.angular.z = angular_speed
            elif straight > threshold_dist and Fleft > threshold_dist and Fright < threshold_dist:
                state_description = 'case 3 - Empodio deksia mprosta'
                speed.linear.x = 0.02
                speed.angular.z = angular_speed
            elif straight > threshold_dist and Fleft < threshold_dist and Fright > threshold_dist:
                state_description = 'case 4 - Empodio eutheia aristera'
                speed.linear.x = 0.02
                speed.angular.z = -angular_speed
            elif straight < threshold_dist and Fleft > threshold_dist and Fright < threshold_dist:
                state_description = 'case 5 - Empodio mprosta kai mprosta deksia'
                speed.linear.x = 0.02
                speed.angular.z = angular_speed
            elif straight < threshold_dist and Fleft < threshold_dist and Fright > threshold_dist:
                state_description = 'case 6 - Empodio mprosta kai mprosta aristera'
                speed.linear.x = 0.02
                speed.angular.z = -angular_speed
            elif straight < threshold_dist and Fleft < threshold_dist and Fright < threshold_dist:
                state_description = 'case 7 - Empodio mprosta kai deksia mprosta kai aristera mprosta'
                speed.linear.x = -0.2
                speed.angular.z = 0.15
            elif straight > threshold_dist and left < threshold_dist and right < threshold_dist:
                state_description = 'case 8 - Empodio mprosta aristera kai mprosta deksia'
                speed.linear.x = 1
                speed.angular.z = 0.1
            else:
                print('imposible')
        else:
            state_description="eftasa_arxi"
            speed.linear.x = 0.0
            speed.angular.z=0.0
            tunnel_state='tunnel_finished'
            k=0
            #print(inc_y, inc_x)
        print(state_description)
        pub.publish(speed)
    pub2.publish(tunnel_state)
    r.sleep()
