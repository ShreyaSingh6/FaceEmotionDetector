import cv2
import DeepFace
# import numpy
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# if not cap.isOpened():
    # cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError("Cannot Open Webcam")
while True:
    ret,frame = cap.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
      face_roi = rgb[y:y+h,x:x+w]
      result = DeepFace.analyze(face_roi, actions=['emotion'],enforce_detection=False)
      emotion=result[0]['dominant_emotion']
      cv2.rectangle(frame,(x, y), (x+w, y+h), (0, 0, 255), 2)
      cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
    cv2.imshow("My Face Detection Project", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()