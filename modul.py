import cv2
import numpy as np
import math


def getLineCoordinates(image):
    denoise = cv2.fastNlMeansDenoisingColored(image,None,40,30,7,21);
    blur = cv2.GaussianBlur(denoise,(5, 5),0);
    edges = cv2.Canny(blur, 50, 150);
    cv2.imshow('edges',edges)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    return lines
    
def getLines(image):
    img = cv2.fastNlMeansDenoisingColored(image.copy(),None,40,30,7,21);
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([0,100,0]), np.array([100,255,255]))
    blue_mask = cv2.inRange(hsv, np.array([70,10,0]), np.array([255,255,255]))
    
    green_line = cv2.bitwise_and(img,img, mask=green_mask)
    blue_line = cv2.bitwise_and(img,img, mask=blue_mask)
    
    green_line = cv2.GaussianBlur(green_line,(5,5),0)
    green_line = cv2.Canny(green_line, 50, 150)
    blue_line = cv2.GaussianBlur(blue_line,(5,5),0)
    blue_line = cv2.Canny(blue_line, 50, 150)
    
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 40  # maximum gap in pixels between connectable line segments
    lines_green = cv2.HoughLinesP(green_line, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    lines_blue = cv2.HoughLinesP(blue_line, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    line_green = sortLine(getLine(lines_green))
    line_blue = sortLine(getLine(lines_blue))
    
    for x1,y1,x2,y2 in line_green:
        for xx1,xx2,yy1,yy2 in line_blue:
            if (y2+y1)/float(2) > (yy2+yy1)/float(2):
                return line_blue, line_green
            else:
                return line_green, line_blue
    
def getLine(lines):
    liness = []
    to_skip = []
    e = 7
    for i in range(0, len(lines)):
        for_avg = []
        if i not in to_skip:
            for x1,y1,x2,y2 in lines[i]:
                for_avg.append(lines[i])
                for j in range(i+1, len(lines)):
                    for xx1,yy1,xx2,yy2 in lines[j]:
                       if abs(x1-xx1) < e and abs(y1-yy1) < e and abs(x2-xx2) < e and abs(y2-yy2) < e:
                           for_avg.append(lines[j])
                           to_skip.append(j)
        else:
            continue
        if len(for_avg) == 1:
            liness.append(for_avg[0])
        else:
            for_avg = np.array(for_avg)
            liness.append((np.mean(for_avg, axis=0).astype(int)))
            
    if len(liness) != 1:
        maxx = 0
        max_line = []
        for x1,y1,x2,y2 in lines[0]:
            maxx = math.hypot(x2 - x1, y2 - y1)
            max_line = np.array([[x1,y1,x2,y2]])
        for i in range(0, len(liness)-1):
            for x1,y1,x2,y2 in liness[i]:
                for xx1,yy1,xx2,yy2 in liness[i+1]:
                    h = math.hypot(x1-xx2, y1 - yy2)
                    if h > maxx:
                        maxx = h
                        max_line = np.array([[x1,y1,xx2,yy2]])    
        liness = [max_line]
    return liness[0]

def sortLine(line):
    for x1,y1,x2,y2 in line:
        if x1 > x2:
            return [[x2, y2, x1, y1]]
        else:
            return line