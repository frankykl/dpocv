#!/usr/bin/env python

import sys,os
from os import listdir
from os.path import isfile, isdir, join
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/../..')

import re
import argparse
import pprint
import time
import datetime
import numpy as np
import cv2

COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0, 0,255)
COLOR_YELLOW = (255,255,0)

"""
Parse input arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description='Visual tracking benchmark')
    parser.add_argument('--data', dest='data_path', help='path of input data', default=None, type=str)
    parser.add_argument('--report', dest='report_path', help='path of report', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


"""
Draw bounding box on an image.
"""
def DrawBBox(im, id, bbox):
    colors = [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[(id + 1) & 3], 2)


def ComputeIOU(box1, box2):
	xA = max(box1[0], box2[0])
	yA = max(box1[1], box2[1])
	xB = min(box1[2], box2[2])
	yB = min(box1[3], box2[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
	boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou    


def RunTestCase(casePath, reportPath=None):
    print("Test case = ", casePath)
    print("Report path = ", reportPath)

    caseName = os.path.basename(os.path.normpath(casePath))
    print("Case name = ", caseName)

    # open report file
    reportFile = None
    if reportPath is not None:
        fileName = os.path.join(reportPath, caseName + '.txt')
        print("reportFile = ", fileName)
        reportFile = open(fileName, 'w')
        reportFile.write("TestCase = %s\n" % casePath)


    # open ground truth file
    gtFile = None
    if os.path.exists(join(casePath, "groundtruth_rect.txt")):
        gtFile = open(join(casePath, "groundtruth_rect.txt"))
    elif os.path.exists(join(casePath, "groundtruth_rect.1.txt")):
        gtFile = open(join(casePath, "groundtruth_rect.1.txt"))
    elif os.path.exists(join(casePath, "groundtruth_rect.2.txt")):
        gtFile = open(join(casePath, "groundtruth_rect.2.txt"))
    else:
        print('ground truth does not exist')
        return

    gt = gtFile.readlines()
    gtFile.close()

    # create background modeler
    #bm = cv2.dp_bgmodel.CreateDPBackgroundModeler()

    # parameters
    numTrackersSet = 1
    keyWaitTime = 1

    # create tracker
    tracker = cv2.dp_tracking.CreateDPMTTracker(numTrackersSet)
    numTrackers = tracker.GetNumberTrackers()
    print("numTrackers =", numTrackers)

    # collect image files
    imagePath = join(casePath, "img")
    imageFiles = [f for f in listdir(imagePath)
            if isfile(join(imagePath, f)) and f.lower().endswith(('.jpg', 'jpeg', '.png'))]
    imageFiles = sorted(imageFiles)
    #print(len(imageFiles))

    # set initial object
    inputImage = cv2.imread(join(imagePath, imageFiles[0]))
    height, width, channels = inputImage.shape
    values = re.split(',| |\t|\n', gt[0])
    det = np.zeros(5, dtype=np.float32)
    det[0] = max(0, float(values[0]))
    det[1] = max(0, float(values[1]))
    det[2] = min(det[0] + float(values[2]), width - 1)
    det[3] = min(det[1] + float(values[3]), height - 1)
    det[4] = 1.0
    tracker.StartTracker(0, inputImage, det)

    # loop for all images
    iouSum = 0.0
    for i in range(1, len(gt)):
        print('Frame = ', i)
        #print(imageFiles[i])

        # process key
        keyId = -1
        key = cv2.waitKey(keyWaitTime) & 0xFF
        if key == ord('q'):
            break

        # read frame
        inputImage = cv2.imread(join(imagePath, imageFiles[i]))
        height, width, channels = inputImage.shape
        display = np.copy(inputImage)
        #print('Input size = ', width, height)

        # read ground truth
        #print(gt[i])
        values = re.split(',| |\t|\n', gt[i])
        x0 = int(values[0])
        y0 = int(values[1])
        x1 = x0 + int(values[2])
        y1 = y0 + int(values[3])

        dets = []
        state = tracker.GetTrackerState(0)
        if state != 1:
            det = np.zeros(5, dtype=np.float32)
            det[0] = max(0, float(values[0]))
            det[1] = max(0, float(values[1]))
            det[2] = min(det[0] + float(values[2]), width - 1)
            det[3] = min(det[1] + float(values[3]), height - 1)
            det[4] = 1.0
            #print('det = ', det)
            dets.append(det)

        # run tracker
        if state == 0:
            tracker.StartTracker(0, inputImage, dets[0])
        else:
            tracker.ProcessFrame(inputImage, np.asarray(dets))

        # draw ground truth
        cv2.rectangle(display, (x0,y0), (x1,y1), COLOR_RED, 2)

        # get tracker results and display
        state = tracker.GetTrackerState(0)
        #print('  state = ', state)
        if state == 1:
            r = tracker.GetTrackerBBox(0)
            DrawBBox(display, 0, r)
            iou = ComputeIOU((x0, y0, x1, y1), (r[0], r[1], r[0] + r[2] - 1, r[1] + r[3] - 1))
        else:
            r = (0, 0, 0, 0)
            iou = 0

        iouSum += iou

        cv2.imshow(casePath, display)

        # report result
        if reportFile is not None:
            reportFile.write("Frame = %d, GT = [%d, %d, %d, %d], Tracker = [%d, %d, %d, %d], IOU = %f\n" %
                    (i, x0, y0, x1, y1, r[0], r[1], r[0] + r[2] - 1, r[1] + r[3] - 1, iou))

    cv2.destroyWindow(casePath)
    cv2.destroyAllWindows()

    if reportFile is not None:
        reportFile.write("MeanIOU = %f" % (iouSum / float(len(gt))))
        reportFile.close()


"""
Run tracker benchmark.
Usage:
python TrackerBenchmark.py --data=... -- report=...
"""
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    rootPath = args.data_path
    reportPath = args.report_path
    
    #RunTestCase(rootPath, reportPath)

    
    testCases = [join(rootPath, o) for o in listdir(rootPath) if isdir(join(rootPath, o))]
    testCases = sorted(testCases)

    for testCase in testCases:
        RunTestCase(testCase, reportPath)
    
