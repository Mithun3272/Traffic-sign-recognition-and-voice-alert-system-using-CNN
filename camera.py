import cv2
import os
if os.path.isfile('image.jpg'):
    os.remove('image.jpg')
vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    #Crop frame to get centered 300x300 
    frame = frame[100:500, 100:500]
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('image.jpg', frame)
        break
vid.release()
cv2.destroyAllWindows()
