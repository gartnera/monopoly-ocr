# import the necessary packages
import numpy as np
import argparse
import cv2
import math
from tesserocr import PyTessBaseAPI, PSM, iterate_level, RIL
from PIL import Image

def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
	deltaY = y_landmark - y_orig
	deltaX = x_landmark - x_orig
	return angle_trunc(atan2(deltaY, deltaX))

def getRotation(box):
	length = len(box)

	minDistance = float("inf")
	minAngle = 0

	for i in range(length):
		curr = box[i]
		if i + 1 == length:
			next = box[0]
		else:
			next = box[i + 1]
		(dx, dy) = (next[0] - curr[0], next[1] - curr[1])
		distance = int(math.hypot(dx, dy))

		angle = math.atan(float(dx)/float(dy))
		angle *= 180/math.pi

		if distance < minDistance:
			minDistance = distance
			minAngle = angle

	return round(minAngle)

def getSize(box):
	length = len(box)
	prevDistance = 0
	for i in range(length):
		curr = box[i]
		if i + 1 == length:
			next = box[0]
		else:
			next = box[i + 1]
		(dx, dy) = (next[0] - curr[0], next[1] - curr[1])
		distance = int(math.hypot(dx, dy))
		if prevDistance == 0:
			prevDistance = distance
		elif prevDistance < distance:
			return (distance, prevDistance)
		else:
			return (prevDistance, distance)

def getCenter(box):
	p1 = box[0]
	p2 = box[2]

	return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

def maskImage(src, dst, mask):
	height, width = src.shape
	for i in range(height):
		for j in range(width):
			if mask[i][j]:
				dst[i][j] = src[i][j]

def classiferChoices(ri):
	level = RIL.SYMBOL
	for r in iterate_level(ri, level):
		symbol = r.GetUTF8Text(level)  # r == ri
		conf = r.Confidence(level)
		if symbol:
			print u'symbol {}, conf: {}'.format(symbol, conf),
		indent = False
		ci = r.GetChoiceIterator()
		for c in ci:
			if indent:
				print '\t\t ',
			print '\t- ',
			choice = c.GetUTF8Text()  # c == ci
			print u'{} conf: {}'.format(choice, c.Confidence())
			indent = True
		print '---------------------------------------------'

def eshow(img):
    cv2.imwhow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
height, width, channels = image.shape

print "Loaded image {}x{}".format(height, width)

lower = np.array([155, 155, 155])
upper = np.array([255, 255, 255])
filtered = cv2.inRange(image, lower, upper)

#kernel = np.ones((1,1),np.uint8)
#filtered = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel)
#kernel = np.ones((3,3),np.uint8)
#filtered = cv2.morphologyEx(filtered,cv2.MORPH_CLOSE, kernel)

im2, contours, hierarchy = cv2.findContours(filtered.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
count = 0

output = np.ones((height,width, 3), np.uint8)
output.fill(255)

currentPosition = 0
maxWidth = 0

for c in contours:
	cLen = cv2.arcLength(c, True)
	#remove small objects
	if cLen < 70:
		cv2.drawContours(filtered, [c], 0, (255,255, 255), -1)

tapi = PyTessBaseAPI(psm=PSM.SINGLE_CHAR)
tapi.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
tapi.SetVariable("save_blob_choices", "T")
for c in contours:
	cLen = cv2.arcLength(c, True)
	if cLen > height / 2:
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		cv2.drawContours(image,[box],0,(0,255,0),5)
		#cv2.circle(filtered, tuple(box[0]), 20, (0,0,255), -1)
		#cv2.circle(filtered, tuple(box[1]), 20, (255,0,0), -1)
		#cv2.circle(filtered, tuple(box[2]), 20, (0,255,255), -1)

		center = getCenter(box)
		rotation = getRotation(box)
		size = getSize(box)
		rotMatrix = cv2.getRotationMatrix2D(center, - rotation , 1.0)
		result = cv2.warpAffine(filtered, rotMatrix, (width, height))
		result = cv2.getRectSubPix(result, size, center)
		#cv2.rectangle(result, (40, 260), (550, 350), (0,255,0))

		result = cv2.getRectSubPix(result, (510, 90), (300,305))
		#cv2.line(result, (0, 37), (510, 37), (0,255,0), 5)

		im2, subContours, hierarchy = cv2.findContours(result.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		mask = np.zeros(result.shape, np.uint8)
		for sc in subContours:
			scLen = cv2.arcLength(sc, True)
			if scLen < 300 and scLen > 100:
				cv2.drawContours(mask, [sc], -1, (255,255,255), -1)
				cv2.drawContours(mask, [sc], -1, (255,255,255), 2)

		mask = mask.astype(np.bool)
		dst = np.zeros(result.shape, np.uint8)
		dst.fill(255)

		maskImage(result, dst, mask)
		np.copyto(dst, result, where=mask)

		#invert and erode
		#dst = (255-dst)
		#kernel = np.ones((5,5),np.uint8)
		#dst = cv2.erode(dst,kernel,iterations = 1)

		dstDrawn = dst.copy()
		dstDrawn = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

		chars = {}
		for sc in subContours:
			scLen = cv2.arcLength(sc, True)
			if scLen < 300 and scLen > 100:

				x, y, w, h = cv2.boundingRect(sc)
				x -= 5
				y -= 5
				w += 10
				h += 10
				center= (x + w / 2, y + h / 2)
				size = (w, h)

				charRect = cv2.getRectSubPix(dstDrawn, size, center)

				start = (x, y)
				end = (x+w, y+h)
				cv2.rectangle(dstDrawn, start, end, (0,255,0), 1)

				rectImg = Image.fromarray(charRect)
				tapi.SetImage(rectImg)
				'''
				tapi.Recognize()
				classiferChoices(tapi.GetIterator())
				continue
				'''
				char = tapi.GetUTF8Text().strip()
				if char:
					cv2.putText(dstDrawn, char, (end[0] - 15, end[1] - 5), cv2.FONT_HERSHEY_COMPLEX, .75, (0,0,255), 2)
				else:
					char = ' '
				chars[x] = char

				'''
				print char

				cv2.imshow("rect", charRect)
				cv2.waitKey(0)
				'''


		codeStr = ''
		for key in sorted(chars):
			codeStr += chars[key]
		print codeStr

		output[currentPosition:currentPosition+result.shape[0], 0:result.shape[1], ] = dstDrawn
		#currentPosition += size[1] + 20
		currentPosition += 100

		if result.shape[1] > maxWidth:
			maxWidth = result.shape[1]

		count += 1

output = output[:currentPosition, :maxWidth]

print "Matched {} pieces".format(count)
cv2.imwrite('new.png', output)
