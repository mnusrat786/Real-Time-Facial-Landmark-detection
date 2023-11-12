import cv2
import mediapipe as mp
import time

video = "your local video address"
cap = cv2.VideoCapture(0)

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)



while True:
    success, img = cap.read() 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for lm in faceLms.landmark:
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                

    cTime = time.time()
    if  cTime-pTime != 0:
        fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"Fps: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)