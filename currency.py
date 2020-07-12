'''
THIS PROGRAM USES SIFT ALGORITHM (SCALE INVARIANT FEATURE TRANSFORM) FOR DETECTING KEYPOINTS AND DESCRIPTORS AND
FLANN FAST LIBRARY FOR APPROXIMATE NEAREST NEIGHBORS BASED MATCHING
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import pyttsx3



engine = pyttsx3.init() #initializing engine

cap=cv2.VideoCapture(0)
#Reading the Video Camera
print('This program detects rs 100, rs 500, rs 200, rs 10 , and 1 dollar notes back and front!')
training_set=[img for img in glob.glob('files/*.jpg')]
#Reading Files
names=[(lambda x: x[6:].strip('.jpg'))(x) for x in glob.glob('files/*.jpg')]
#Reading Names
l=[0]*(len(training_set)) 
#List Assignment
sift = cv2.xfeatures2d.SIFT_create()
#Scale Invariant Feature Transform
ret, test_img=cap.read()
while True:
    
    cv2.imshow('frame',test_img)
    ret, test_img=cap.read()
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
    j=0
    kp1, des1 = sift.detectAndCompute(test_img,None)
    while j<len(training_set):
        train=cv2.imread(training_set[j])
        kp2, des2 = sift.detectAndCompute(train,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=60) 

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des1,des2,k=2)
        matchesMask = [[0,0] for i in range(len(matches))]
        #ratio testing
        #The distance ratio between the two nearest matches of a considered keypoint is computed and it is a good match when this value is below a thresold. 
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]

        l[j]=(np.sum(matchesMask))
        j=j+1
 
    temp2=l[:]
    l.sort()
    if l[len(l)-1]-l[len(l)-2]>15:
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
        
        img3 = cv2.drawMatchesKnn(test_img,kp1,cv2.imread(training_set[temp2.index(l[len(l)-1])]),kp2,matches,None,**draw_params)

        plt.imshow(img3,),plt.show()

        cv2.putText(test_img,names[temp2.index(l[len(l)-1])], (100,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2);
        print("Detected Denomination "+names[temp2.index(l[len(l)-1])])
        engine.say("this is")
        engine.say(names[temp2.index(l[len(l)-1])])
        engine.runAndWait()
cap.release()
cv2.destroyAllWindows()
