#!/usr/bin/env python3

import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
import rospy 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import math
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
decision='lane_detection'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
labels = [{'name':'paking', 'id':1}, {'name':'stop', 'id':2}, {'name':'tunnel', 'id':3}]
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')




configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-7')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

def image_callback(frame):
    
    bridge = CvBridge()
    try:
        img= bridge.imgmsg_to_cv2(frame,"bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

        max_boxes_to_draw = boxes.shape[0]
        scores = detections['detection_scores']
        min_score_thresh=.8
        global decision
        eidos=detections['detection_classes']+label_id_offset
        decision='lane_detection'
        r = rospy.Rate(1)
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                #print ("This box is gonna get used", boxes[i])
                x1=int(boxes[i][1]*320)
                y1=int(boxes[i][0]*240)
                x2=int(boxes[i][3]*320)
                y2=int(boxes[i][2]*240)
                p1=[x1,y1]
                p2=[x2,y2]
                #print(x1,y1,x2,y2)
                if eidos[i]==1:
                    diametros = int(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ))
                    print("Parking sign detected")
                    if diametros > 110:
                        decision='parking_process'
                        print('parking process starting now...')
                        pub2.publish(decision)
                        print(decision)
                        r.sleep()  
                if eidos[i]==2:
                    diametros = int(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ))
                    print("Stop sing detected",int(diametros))
                    #if diametros> 90:
                    decision='stoping_process'
                    pub2.publish(decision)
                    print(decision)
                    #r.sleep()  
                if eidos[i]==3:
                    diametros = int(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ))
                    print("tunnel sign detected")
                    if diametros > 78:
                        decision='tunnel_process'
                        print('Tunnel process starting now...')
                        pub2.publish(decision)
                        print(decision)
                        r.sleep()  
                else:
                    decision='lane_detection'
                    print(decision)
                #r.sleep()
rospy.init_node('sign_detection')
sub=rospy.Subscriber('/camera/image',Image,image_callback)             
pub2 = rospy.Publisher('/chater', String,queue_size=5)
r = rospy.Rate(5)
while not rospy.is_shutdown():

    #print(decision)   
    r.sleep()
        #pub=rospy.Publisher('/cmd_vel', Twist)
        #move=Twist()
    #rospy.spin()