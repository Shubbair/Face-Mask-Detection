import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import datetime

# capture video from local camera
cap = cv2.VideoCapture(0)

# load face detection model
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# load training facemask model
model = tf.keras.models.load_model('models/facemask_detection.h5')
model.summary()

while cap.isOpened():
    _,img = cap.read()
    face = face_cascade.detectMultiScale(img,scaleFactor = 1.1,minNeighbors = 5)
    for(x,y,w,h) in face:
        face_img = img[y:y+h,x:x+w]

        cv2.imwrite('temp.jpg',face_img)
        
        # process the image before feed it to the model
        image = tf.keras.utils.load_img('temp.jpg', target_size=(150, 150))
        image = tf.image.rgb_to_grayscale(image)
        image = tf.keras.utils.img_to_array(image) / 255.0

        image_flatten = image.reshape((-1, 150, 150,1))

        pred = model.predict(image_flatten,batch_size=128)[0]
        (mask, withoutMask) = pred
        
        # if there is no mask
        if mask < withoutMask:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),3)

        # if there is a mask
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(50,255,50),3)

        cv2.imshow('Image',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
