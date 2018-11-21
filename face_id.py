import os
import cv2
import numpy as np
import time
from PIL import Image


def detector():
    face_cascade = cv2.CascadeClassifier('recognizer/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer/trainingData.yml")
    id = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # ID:NAME
    id_list = {}
    with open("face_to_id.txt") as file:
        for line in file:
            id_list[int(line.split(":")[0])] = str(line.split(":")[1].split("\n")[0]) # just to get rid of the newline char
            
    while True:
        ret,rgb_frame = cap.read()
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
        
        for (x,y,w,h) in faces:
            roi = gray_frame[y:y+h,x:x+w]
            cv2.rectangle(rgb_frame,(x,y),(x+w,y+h),(255,0,0),1)
            
            id,conf = rec.predict(roi)
            if conf > 50:
                name = id_list[id] + str(conf)
            else:
                name = "Unknown" + str(conf)
                
            cv2.putText(rgb_frame,name,(x,y+h),font,2,(0,255,0),2)
            
        cv2.imshow("Camera",rgb_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

## Will search the entirty of the subject folder putting all the faces
## together into one training file.
def getImageWithID(name,path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    faces=[]
    IDs=[]
    
    for imagePath in imagePaths:
        print(imagePath)
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        faces.append(faceNp)
        id = imagePath.split("-")[1]
        IDs.append(int(id))
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return IDs, faces
    
    
def recognize(ids,faces):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()


def data_collector(name,path,userID):
    face_cascade = cv2.CascadeClassifier('recognizer/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    pic_num = 1
    
    while pic_num <= 20:
        try:
            ret, rgb_frame = cap.read()
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
            
            if (faces != ()):
                for (x,y,w,h) in faces:
                    roi = gray_frame[y:y+h,x:x+w]
                    resized_image = cv2.resize(roi,(100,100))
                    cv2.imwrite(path+name+"-"+str(userID)+"-"+str(pic_num)+'.jpg',resized_image)
                    
   
            time.sleep(.25)
            pic_num += 1
            cv2.imshow("Frame",rgb_frame)
        except Exception as e:
            print(str(e))
            
    cap.release()
    cv2.destroyAllWindows()
        
    
if __name__ == "__main__":

    ans = 0
    print("""Options:
        1) Detect
        2) Add new user
        3) Exit""")
    ans = int(input("Choose: "))

    while (ans!=0):
        if ans == 1:
            detector()
            ans = 0
        elif ans == 2:
            name = input("Name of subject: ")
            userID = input("User ID: ")
            path = "subject/" # path to images of subject
            if not os.path.exists(path):
                os.makedirs(path)
            
            with open("face_to_id.txt","a") as file:
                file.write(str(userID)+":"+str(name))
            
            data_collector(name,path,userID)
            IDs,faces = getImageWithID(name,path)
            recognize(IDs,faces)
            ans = 1
        else:
            ans = input("Try again: ")