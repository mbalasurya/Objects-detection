<!DOCTYPE>  
<html>  
<head>   
<body>
<h1>Object Detection</h1>
<p>Detecting an is classified by the of the pixels in the image.


<img src="![download](https://github.com/mbalasurya/Objects-detection/assets/110713151/79dbb8bd-0d3e-4886-895c-234d3ebb54ef)">

The image is segmented or classified.And the classified pixels are denotedin boxes by the coverage of boxes the object is identified.
OpenCV is a huge open-source library for computer vision, machine learning, and image processing. OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. 

It can process images and videos to identify objects, faces, or even the handwriting of a human. When it is integrated with various libraries, such as Numpy which is a highly optimized library for numerical operations, then the number of weapons increases in your Arsenal i.e whatever operations one can do in Numpy can be combined with OpenCV.

This OpenCV tutorial will help you learn the Image-processing from Basics to Advance, like operations on Images, Videos using a huge set of Opencv-programs and projects.

OpenCV is the huge open-source library for computer vision, machine learning, and image processing and now it plays a major role in real-time operation which is very important in today's systems. By using it, one can process images and videos to identify objects, faces, or even the handwriting of a human.</p>
<p>import cv2
import pandas

img =cv2.imread("download (1).jpg")


import matplotlib.pyplot as plt


config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'
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
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img, classlabels[classInd-1], (boxes[0]+10, boxes[1]+40), font,fontScale=font_scale, color=(0, 255, 0), thickness=3)
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))</p>



</body>  
</head>  
</html>  
