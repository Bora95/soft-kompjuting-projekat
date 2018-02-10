import cv2
import keras
import numpy as np
from collections import namedtuple
import os
Digit = namedtuple('Digit', 'rect digit prob')

class NumberFinder:
    __model = None
    def __init__(self, model_name = 'model.h5'):
        try:
            self.__model = keras.models.load_model(model_name)
        except Exception:
            print "Not valid name: " + model_name
            self.__model = None

    def __process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,0])
        upper = np.array([255,70,255])
        mask = cv2.inRange(hsv, lower, upper)
        im_msk = cv2.bitwise_and(img,img, mask=mask)
        im_grey = cv2.cvtColor(im_msk, cv2.COLOR_BGR2GRAY)
        return im_grey

    def getNumbers(self, img, currFrame):
        ret = []
        im_p = self.__process(img)
        im_d = im_p.copy()
        im2, ctrs, hier = cv2.findContours(im_p.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        for rect in rects:
            if rect[3] < 15:
                continue
            leng = int(rect[3] * 1.1)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_p[pt1:pt1+leng, pt2:pt2+leng]
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = roi.reshape(1, 28, 28, 1).astype('float32') / 255
            #roi = roi.reshape(1, 1, 28, 28).astype('float32') / 255
            nbr = self.__model.predict_classes(roi)
            prob = self.__model.predict_proba(roi)
            if(prob[0][nbr[0]] > 0.9):
                cv2.putText(im_d, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (200, 200, 200), 3)
                ret.append(Digit(rect = rect, digit = nbr[0], prob = prob[0].tolist()))
        cv2.imwrite(os.path.join('images', 'image'+`currFrame`+'.jpg'), im_d)
        return ret
