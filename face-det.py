__author__ = 'Briana'
import cv2, time, cv2.cv as cv
import numpy as np

path = 'OpenCV\opencv\data\haarcascades\\'
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(path+'haarcascade_eye.xml')

def detect(img, cascade=face_cascade):
    rects = cascade.detectMultiScale(img)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def view(img):
    img_color = img     #frame from camera
    img_gray = cv2.cvtColor(img_color, cv.CV_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)

    faces = detect(img_gray)
    img_out = img_color.copy()
    for face in faces:  #mark faces with rects
        draw_rects(img_out, faces, (0, 255, 0))
        eyes = detect(img_gray, cascade=eye_cascade)
        for eye in eyes:    #mark eyes for each face with rects
            draw_rects(img_out, eyes, (0, 0, 255))
    return img_out      #return img with rectangles for display

if __name__ == "__main__":
    cv2.namedWindow("Live Feed")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        cond, img = vc.read()
    while cond:
        cond, img = vc.read()
        key = cv2.waitKey(20)
        if key == 27: #ESC to exit
            break
        cv2.imshow("Live Feed", view(img))

    vc.release()
    cv2.destroyWindow("Live Feed")
