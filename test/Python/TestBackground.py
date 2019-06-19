import sys
import numpy as np
import cv2

#print(cv2)

# Function to analyze motion energy
def AnalyzeMotion(motionMap):
    thr = 16
    th, motionMask = cv2.threshold(motionMap, thr, 128, cv2.THRESH_BINARY)
    #cv2.imshow("motionMask", motionMask)
    
    motionMap[motionMask == 0] = 0
    pixelCount = cv2.countNonZero(motionMap)
    motionEnergy = 0
    if pixelCount != 0:
        motionEnergy = np.sum(motionMap ** 2) / pixelCount
    print(motionEnergy)


def main():
    # open video capture
    cap = cv2.VideoCapture(sys.argv[1])

    # create background modeler
    bm = cv2.dp_bgmodel.CreateDPBackgroundModeler()

    while (cap.isOpened()):
        # get a video frame
        ret, frame = cap.read()
        if not ret:
            break

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if width > 1280:
            frame = cv2.resize(frame, (1280, 720))

        # feed frame to background modeler
        bm.ProcessFrame(frame)

        cv2.imshow("Input", frame)

        parvo = bm.GetParvoOutput()
        motionMap = bm.GetMotionMap()
        #motionMask = bm.GetMotionMask()
        background = bm.GetBackgroundFrame()
        foreground = bm.GetForegroundFrame()

        #AnalyzeMotion(motionMap)

        #th1 = cv2.adaptiveThreshold(motionMap, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
        #th1 = cv2.adaptiveThreshold(motionMap, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
        #ret, th1 = cv2.threshold(motionMap, 16, 255, cv2.THRESH_BINARY)
        #cv2.imshow('th1', th1)

        '''
        kernel3 = np.ones((3,3),np.uint8)
        kernel5 = np.ones((5,5),np.uint8)
        kernel7 = np.ones((7,7),np.uint8)
        th1 = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel3)
        th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel5)
        cv2.imshow('Morph', th1)

        im2, contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        height, width = th1.shape
        display = cv2.resize(frame, (width, height))
        cv2.drawContours(display, contours, -1, (0,255,0), 3)
        cv2.imshow('contour', display)
        '''

        '''
        edge = cv2.Canny(parvo, 100, 300)
        cv2.imshow("Edge", edge)
        '''

        '''
        #sift = cv2.xfeatures2d.SIFT_create()
        #kp = sift.detect(motionMap, None)
        surf = cv2.xfeatures2d.SURF_create(2000)
        kp, des = surf.detectAndCompute(motionMap, None)
        kpImg = cv2.drawKeypoints(motionMap, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("kp", kpImg)
        '''

        # display results
        cv2.imshow('Parvo', parvo)
        cv2.imshow('Motion Map', motionMap)
        #cv2.imshow('Motion Mask', motionMask)
        #cv2.imshow('Background', background)
        #cv2.imshow('Foreground', foreground)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
