import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for nb, lm in enumerate(handLms.landmark):
                print(nb, lm)
                h, w, c = img.shape
                cx, cy = int(w*lm.x), int(h*lm.y)
                print(nb, cx, cy)
                cv2.putText(img, str(nb), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # if nb == 4:
                #     cv2.circle(img, (cx, cy), 15, (0, 0, 0), -1)
                # if nb == 20:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 0), -1)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)



    cv2.imshow("Video", img)
    if cv2.waitKey(0) == ord('q'):
        break