
"""
Created on Monday NOV 9 14:55:19 2026

@author: GIO
"""

import numpy as np
import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("contornos", cv2.WINDOW_NORMAL)
cv2.namedWindow("umbral", cv2.WINDOW_NORMAL)

while True:
    res, frame = cam.read()
    img = frame.copy()
    ajustada = cv2.convertScaleAbs(img, alpha=1.2, beta=-80)
    gray=cv2.cvtColor( ajustada,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(gray, 215,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    """th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,-2)"""
    kernel=np.ones((5,5),np.uint8)
    closingImg=cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contornos,_=cv2.findContours(closingImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    Lista=[]

    for index in range(len(contornos)):
        area = cv2.contourArea(contornos[index])
        if area > 10000:
            cnt = contornos[index]
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            cv2.putText(ajustada, str(area), (int(x),int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            color = (0,255,0)
            Lista.append(cnt)
            if area < 29500:
                color=(0,0,255)
                cv2.putText(ajustada, "Defectuosa", (int(x)+100,int(y)+100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                color=(0,255,255)


    cv2.drawContours(ajustada, Lista, -1,color,1)

    cv2.imshow("contornos",  ajustada)
    cv2.imshow("umbral", th)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


        