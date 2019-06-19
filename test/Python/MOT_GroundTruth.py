"""
This program reads MOT video and ground truth and generate training samples.

Usage:
python PETS_GroundTruth.py .../MOT2015/train/PETS09-S2L1
"""

import sys, os
import numpy as np
import cv2

#print(cv2)

Colors = [(255,0,0), (0,0,255), (0,255,0), (255,255,0), (0,255,255), (128,128,0), (0,128,128), (128,0,128)]

"""
Draw objects on a frame.
"""
def DrawObjects(frame, objects):
	#print len(objects)
	#print objects
	for obj in objects:
		objId = obj[0]
		color = Colors[objId & 7]
		x0 = int(obj[1])
		y0 = int(obj[2])
		x1 = int(obj[1] + obj[3] + 0.5)
		y1 = int(obj[2] + obj[4] + 0.5)
		cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

"""
Check IoU between negative sample and detected objects.
"""
def CheckIoU(x0, y0, x1, y1, objects, thr):
	for obj in objects:
		obj_x0 = int(obj[1])
		obj_y0 = int(obj[2])
		obj_x1 = int(obj[1] + obj[3] + 0.5)
		obj_y1 = int(obj[2] + obj[4] + 0.5)

		ix0 = max(x0, obj_x0)
		iy0 = max(y0, obj_y0)
		ix1 = min(x1, obj_x1)
		iy1 = min(y1, obj_y1)
		iarea = (ix1 - ix0) * (iy1 - iy0)

		ux0 = min(x0, obj_x0)
		uy0 = min(y0, obj_y0)
		ux1 = max(x1, obj_x1)
		uy1 = max(y1, obj_y1)
		uarea = (ux1 - ux0) * (uy1 - uy0)

		if iarea > uarea * thr:
			return False

	return True

"""
Generate a negative sample.
"""
def GenerateNegativeSample(frame, objects, width, height):
	import random
	imageHeight, imageWidth, imageChannels = frame.shape

	while (1):
		x0 = random.randint(1, imageWidth - width)
		y0 = random.randint(1, imageHeight - height)
		if CheckIoU(x0, y0, x0 + width, y0 + height, objects, 0.5):
			negImage = frame[y0:y0 + height, x0:x0 + width]
			negImage = cv2.cvtColor(negImage, cv2.COLOR_BGR2GRAY)
			negImage = cv2.resize(negImage, (64, 128))
			break

	return negImage


"""
Generate positive and negative samples based on ground truth.
"""
def GenerateTrainingSamples(frameId, frame, objects, samplePath):
	imageHeight, imageWidth, imageChannels = frame.shape
	posSamples = []
	negSamples = []

	# loop for objects
	for obj in objects:
		objId = obj[0]
		color = Colors[objId & 7]
		scaleH = 0.1
		scaleV = 0.08
		x0 = max(int(obj[1] - obj[3] * scaleH), 0)
		y0 = max(int(obj[2] - obj[4] * scaleV), 0)
		x1 = min(int(obj[1] + obj[3] * (1 + scaleH) + 0.5), imageWidth)
		y1 = min(int(obj[2] + obj[4] * (1 + scaleV) + 0.5), imageHeight)
		width = x1 - x0
		height = y1 - y0
		#cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

		if width >= 16 and height >= 16:
			# positive sample file name
			posFile = "pos_" + str(frameId).zfill(6) + "_" + str(objId) + ".jpg"
			#print(posFile)
			posSamples.append(posFile)

			# crop and scale sample
			sampleWidth = 64
			sampleHeight = 128
			posImage = frame[y0:y1, x0:x1]
			posImage = cv2.cvtColor(posImage, cv2.COLOR_BGR2GRAY)
			posImage = cv2.resize(posImage, (64, 128))
			cv2.imwrite(os.path.join(samplePath, posFile), posImage)

			# display positive sample
			posWindow = str(objId) + 'pos' 
			cv2.imshow(posWindow, posImage)
			cv2.moveWindow(posWindow, objId * 70, 30)

			# negative sample file name
			negFile = "neg_" + str(frameId).zfill(6) + "_" + str(objId) + ".jpg"
			#print(negFile)
			negSamples.append(negFile)

			# create negative sample
			negImage = GenerateNegativeSample(frame, objects, width, height)
			cv2.imwrite(os.path.join(samplePath, negFile), negImage)

			# display negative sample
			negWindow = str(objId) + 'neg' 
			cv2.imshow(negWindow, negImage)
			cv2.moveWindow(negWindow, objId * 70, 230)

	return posSamples, negSamples



"""
Main function flow.
"""

# file paths
sourcePath = sys.argv[1]
gtPath = os.path.join(sourcePath, 'gt')
imagePath = os.path.join(sourcePath, 'img1')
videoFileName = os.path.join(imagePath, "video.mp4")
gtFileName = os.path.join(gtPath, 'gt.txt')
samplePath = os.path.join(sourcePath, 'samples')

print("Source path = ", sourcePath)
print("Video file = ", videoFileName)
print("GT file = ", gtFileName)
print("Sample path = ", samplePath)

# create sample folder if not exist
if not os.path.exists(samplePath):
	os.makedirs(samplePath)

# open gt file
gtFile = open(gtFileName, 'r')

posList = []
negList = []
frameId = 1
while (1):
	print('frame', frameId)

	inputName = str(frameId).zfill(6) + '.jpg'
	fileName = imagePath + '/' + inputName
	#print(fileName)

	# read input frame
	if os.path.isfile(fileName):
		frame = cv2.imread(fileName)
	else:
		break

	# read gt for current frame
	objects = []
	while True:
		lastPos = gtFile.tell()
		line = gtFile.readline()
		if len(line) < 10:
			break
		data = line.split(",")
		if int(data[0]) != frameId:
			gtFile.seek(lastPos)
			break
		obj = [int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])]
		objects.append(obj)

	# draw objects
	#DrawObjects(frame, objects)

	# generate training samples for one image
	posSamples, negSamples = GenerateTrainingSamples(frameId, frame, objects, samplePath)
	posList.extend(posSamples)
	negList.extend(negSamples)

	cv2.imshow("Input", frame)

	frameId += 1

	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()

# write sample list file
#print(sampleList)
print("Positive samples = ", len(posList))
print("Negative samples = ", len(negList))

listFile = open(os.path.join(samplePath, 'pos_list.txt'), 'w')
for item in posList:
	listFile.write("%s\n" % item)
listFile.close()

listFile = open(os.path.join(samplePath, 'neg_list.txt'), 'w')
for item in negList:
	listFile.write("%s\n" % item)
listFile.close()

listFile = open(os.path.join(samplePath, 'sample_list.txt'), 'w')
for item in posList:
	listFile.write("%s, %d\n" % (item, 1))
for item in negList:
	listFile.write("%s, %d\n" % (item, 0))
listFile.close()
