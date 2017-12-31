import cv2

input_image = cv2.imread("../data/me.jpg")

gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(gray, 1.05, 3, cv2.CASCADE_SCALE_IMAGE,(50,50))

