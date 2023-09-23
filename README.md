<!DOCTYPE>  
<html>  
<head>   
<body>
<h1>Object Detection using intel libraries</h1>
<p>Detecting an is classified by the of the pixels in the image.</p>

<p>OpenCV (Open Source Computer Vision) is a library of programming functions for real time computer vision, and Intel® IPP provides the optimized functions for the OpenCV functions.</p>

<p>The image is segmented or classified.And the classified pixels are denotedin boxes by the coverage of boxes the object is identified.
OpenCV is a huge open-source library for computer vision, machine learning, and image processing. OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. </p>

<p>It can process images and videos to identify objects, faces, or even the handwriting of a human. When it is integrated with various libraries, such as Numpy which is a highly optimized library for numerical operations, then the number of weapons increases in your Arsenal i.e whatever operations one can do in Numpy can be combined with OpenCV.</p>

<p>This OpenCV tutorial will help you learn the Image-processing from Basics to Advance, like operations on Images, Videos using a huge set of Opencv-programs and projects.</p>

<p>OpenCV is the huge open-source library for computer vision, machine learning, and image processing and now it plays a major role in real-time operation which is very important in today's systems. By using it, one can process images and videos to identify objects, faces, or even the handwriting of a human.</p>

<h3>#importing cv2</h3>

<p>import cv2

<h3>#importing pyipp as ipp</h3>

<p>import pyipp as ipp</p>
    
<h3>#importing pandas</h3>

<p>import pandas</p>

<h3>#loading the image</h3>

<p>img =cv2.imread("download (1).jpg")</p>

<h3>#importing matplotlib as plt</h3>

import matplotlib.pyplot as plt

<h3>#By using config_file and frozen_inference the object is detected by mobile_net</h3>
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

<h3>#After detect the model and inputs</h3>

model=cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

classIndex ,confidence, bbox = model.detect(img)
print(classIndex)


classlabels=[]
file_name='labels.txt'
with open(file_name,'rt') as fpt:
    classlabels=fpt.read().rstrip('\n').split('\n')

font_scale=2
font=cv2.FONT_HERSHEY_PLAIN
for classInd,Conf,boxes in zip(classIndex.flatten(),confidence.flatten(),bbox):
    
<h3>#importing pyipp</h3>

<p>import pyipp as ipp</p> 
    
cv2.rectangle(img,boxes,(255,0,0),2)
cv2.putText(img, classlabels[classInd-1], (boxes[0]+10, boxes[1]+40), font,fontScale=font_scale, color=(0, 255, 0), thickness=3)
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
<p>
import cv2
import numpy as np
from openvino.inference_engine import IECore

# Path to the OpenVINO model files and labels
model_xml = 'pedestrian-detection-adas-0002.xml'
model_bin = 'pedestrian-detection-adas-0002.bin'
labels_file = 'labels.txt'

# Load the Inference Engine
ie = IECore()

# Load the model
net = ie.read_network(model=model_xml, weights=model_bin)

# Get the input and output names
<p>input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
cap=cv2.VideoCapture("traffic.mp4")</p>


<p>if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
    font_scale=3
    font = cv2.FONT_HERSHEY_PLAIN</p>
<p></p>while True:
    ret, frame = cap.read()
    config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model='frozen_inference_graph.pb'</p>
    <p>model=cv2.dnn_DetectionModel(frozen_model,config_file)
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)</p>
    <p>classIndex, confidence, bbox= model.detect(frame)
    print(classIndex)
if (len(classIndex)!=0):
    for classInd, conf, boxes in zip(classIndex. flatten(), confidece.flatten(), bbox):
        if (ClassInd<=80):
            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
            cv2.putText(frame,classLabels [ClassInd-1], (boxes[0]+18, boxes[1]+48), font, fontScale-font_scale, color=(0, 255, 0), thickness=3)
        cv2.imshow("Object Detection Tutorial", frame)
        if cv2.waitKey(2) & 0xFF== ord('q'):
            break
cap.release()
cv2.destroyAllWindows()</p>

</p>
</body>  
</head>  
</html>  
