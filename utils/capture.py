import cv2
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


def VideoWrite():
    try:
        cap = cv2.VideoCapture(0)
    except:
        return

    width = int(cap.get(3))
    height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter('./record.mp4', fourcc, 10, (width, height))

    while(True):
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('video', frame)
        out.write(frame)

        k = cv2.waitKey(1)
        if(k == 27):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


VideoWrite()
