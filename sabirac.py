import cv2
import modul
import sys
import math
import numpy as np
from number_finder import NumberFinder
from collections import namedtuple
FrameDigit = namedtuple('FrameDigit', 'F D')
DigitFit = namedtuple('DigitFit', 'digit start end fit startFrame')
Point = namedtuple('Point', 'x y')

def onSegment(p, q, r):
    if (q[0] <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
       return True
    return False

def orientation(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0: 
        return 0
    if val > 0: 
        return 1 
    else: 
        return 2

def doIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2);
    o2 = orientation(p1, q1, q2);
    o3 = orientation(p2, q2, p1);
    o4 = orientation(p2, q2, q1);
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and onSegment(p1, p2, q1): return True
    if (o2 == 0 and onSegment(p1, q2, q1)): return True
    if (o3 == 0 and onSegment(p2, p1, q2)): return True
    if (o4 == 0 and onSegment(p2, q1, q2)): return True
    return False


def pastLine(line, x):
    for x1, y1, x2, y2 in line:
        v1 = [x2-x1, y2-y1]   # Vector 1
        v2 = [x2-x[0], y2-x[1]]   # Vector 1
        xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product
        if xp > 0:
            return False
        else:
            return True

def onLine(line, x):
    for x1, y1, x2, y2 in line:
        if x[0] >= x1 and x[0] <= x2:
            if y1 > y2:
                if x[1] <= y1 and x[1] >= y2:
                    return True
                else:
                    return False
            else:
                if x[1] >= y1 and x[1] <= y2:
                    return True
                else:
                    return False
        else:
            return False

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return False
    
def checkPointInSector(center, radius, point, startAngle, endAngle):
    x = point[0] - center[0]
    y = point[1] - center[1]
    startAngle = (math.pi*startAngle)/180
    endAngle = (math.pi*endAngle)/180
    polarradius = math.sqrt(x*x+y*y)
    Angle = math.atan2(y,x)
    if Angle>=startAngle and Angle<=endAngle and polarradius<radius:
        return True
    else:
        return False

def addBorder(img):
    color = [0, 0, 0]
    top, bottom, left, right = [30]*4
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def findNext(frames ,digit, startFrame, e, it):
    for i in range (0, it):
        if startFrame + i <= len(frames):
            frame = frames[startFrame + i]
            x1 = digit.rect[0]
            y1 = digit.rect[1]
            for j in range(0, len(frame)):
                x2 = frame[j].rect[0]
                y2 = frame[j].rect[1]
                if (frame[j].digit == digit.digit) and (math.hypot(x1-x2, y1-y2) < e * (i+1)) and (y2 >= y1 or x2 >= x1):
                #if (frame[j].digit == digit.digit) and checkPointInSector((x1,y1), e * (i+1), (x2,y2), -10, 100):
                    return startFrame + i, j
    return -1, -1

def getCenter(rect):
    rect = np.array(rect, 'float')
    return rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0

def getSum(video, model):
    nf = NumberFinder(model) #model name
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    frame = addBorder(frame)
    img = frame.copy()
    first, second  = modul.getLines(frame)
    frames = {}
    f = open('log.txt', 'w')
    
    currFrame = 0
    while(ret):
        currFrame += 1
        #print currFrame
        frames[currFrame] = nf.getNumbers(frame, currFrame)
        ret, frame = cap.read()
        frame = addBorder(frame)
    cap.release()
    
    extracted_digits = []
    
    for key in frames:
        for digit in frames[key]:
            f.write(`key`+':'+`digit` + '\n')
        f.write('\n')
    
    frame_it1 = 1
    e = 5
    it = 30
    while frame_it1 < len(frames):
        digit_it1 = 0
        while digit_it1 < len(frames[frame_it1]):
            frame_it2, digit_it2 = findNext(frames, frames[frame_it1][digit_it1], frame_it1 + 1, e, it)
            if frame_it2 == -1:
                digit_it1 += 1
            else:
                digit = frames[frame_it1].pop(digit_it1)
                extracted_digits.append([FrameDigit(frame_it1, digit)])
                while frame_it2 != -1:
                    digit = frames[frame_it2].pop(digit_it2)
                    extracted_digits[len(extracted_digits)-1].append(FrameDigit(frame_it2, digit))
                    frame_it2, digit_it2 = findNext(frames, digit, frame_it2 + 1, e, it)
        frame_it1 += 1
    
    f.write('--\n')
    for digits in extracted_digits:
        for digit in digits:
            f.write(`digit` + '\n')
        f.write('--\n')
    f.close()
    
    fits = []
    for digits in extracted_digits:
        if len(digits) > 80:
            xx = []
            yy = []
            for digit in digits:
                x, y = getCenter(digit.D.rect)
                xx.append(x)
                yy.append(y)
            p = np.polyfit(xx, yy, 1)
            p = np.poly1d(p)
            fits.append(DigitFit(digits[0].D.digit, Point(int(xx[0]), int(p(xx[0]))), Point(int(xx[len(xx)-1]), int(p(xx[len(xx)-1]))), p, digits[0].F))
            #cv2.line(img, (int(xx[0]), int(p(xx[0]))), (int(xx[len(xx)-1]), int(p(xx[len(xx)-1]))), (0,255,0), 1)
    sys.stdout.flush()
    add = []
    sub = []
    for fit in fits:
        for x1, y1, x2, y2 in first:
            if doIntersect(fit.start, fit.end, Point(x1, y1), Point(x2, y2)):
                #print fit.digit
                cv2.line(img, fit.start, fit.end, (0,255,0), 1)
                add.append(fit.digit)
            #elif fit.startFrame > 20:
                #ints = intersection(line(fit.start, fit.end), line(Point(x1, y1), Point(x2, y2)))
                #if ints and onLine(first, ints) and pastLine(first, fit.start):
                    #cv2.line(img, fit.start, fit.end, (0,255,0), 1)
                    #add.append(fit.digit)

        for x1, y1, x2, y2 in second:
            if doIntersect(fit.start, fit.end, Point(x1, y1), Point(x2, y2)):
                #print -fit.digit
                cv2.line(img, fit.start, fit.end, (0,255,0), 1)
                sub.append(-fit.digit)
            #elif fit.startFrame > 20:
                #ints = intersection(line(fit.start, fit.end), line(Point(x1, y1), Point(x2, y2)))
                #if ints and onLine(second, ints) and pastLine(second, fit.start):
                    #cv2.line(img, fit.start, fit.end, (0,255,0), 1)
                    #add.append(fit.digit)
            
    print add
    print sub
    return sum(add) + sum(sub), img

#ret = getSum('video-0.avi')
#print ret[0]
#cv2.imshow('',ret[1])

#cv2.waitKey(0)
#cv2.destroyAllWindows()
#for i in range(0,20):
#    cv2.waitKey(1)
