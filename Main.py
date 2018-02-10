import sabirac
import cv2

video = ["video-0.avi","video-1.avi","video-2.avi","video-3.avi","video-4.avi","video-5.avi","video-6.avi","video-7.avi","video-8.avi","video-9.avi"]
video1 = ['video-0.avi']
f = open('out.txt', 'w')
f.write("RA 153/2014 Borislav Puzic\nfile\tsum\n")
for v in video:
    print v
    suma, img = sabirac.getSum(v, 'model.h5')
    print suma
    f.write(v+'\t' + `suma` + '\n')
f.close()

cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(0,30):
    cv2.waitKey(1)