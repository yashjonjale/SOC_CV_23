import cv2
import numpy as np
model_path="/media/yashjonjale/Elements/My_workspace/SOC_CV_23/object_detection/best2.onnx"
path="/media/yashjonjale/Elements/My_workspace/SOC_CV_23/object_detection/demo_images/img"
imgl=[]
for i in range(1,4):
    imgl.append(path+str(i)+".jpg")
net = cv2.dnn.readNetFromONNX(model_path)
file = open("/media/yashjonjale/Elements/My_workspace/SOC_CV_23/object_detection/coco1.txt","r")
classes = file.read().split('\n')
print(classes)

for i in imgl:
    img = cv2.imread(i)
    if img is None:
        break
    img = cv2.resize(img, (1000,600))
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
  

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5:#
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)

    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),1)
        cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),1)

    cv2.imshow("img",img)
    k = cv2.waitKey(10000)
    if k == ord('q'):
        break