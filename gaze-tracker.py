#!/usr/bin/env python2.7

"""
gaze-tracker
by Brandon Jackson and Kerry Clavadetscher

gaze-tracker.py
Main python script
"""

# Import Libraries
import time
import math
from collections import deque
import cProfile

import numpy as np
import cv2
import cv2.cv as cv
import Image
import ImageOps
import ImageEnhance
#from scipy.cluster import vq
#import wx
#import matplotlib
#import matplotlib.pyplot as plt

 
# Constants
#CAMERA_INDEX = 0;
SCALE_FACTOR = 5; # video size will be 1/SCALE_FACTOR
FACE_CLASSIFIER_PATH = "classifiers/haar-face.xml";
EYE_CLASSIFIER_PATH = "classifiers/haar-eyes.xml";
FACE_MIN_SIZE = 0.2;
EYE_MIN_SIZE = 0.03;

DISPLAY_SCALE = 0.3333;
FACE_SCALE = 0.25;
EYE_SCALE = 0.33333;

GAZE_RADIUS = 15;


class FaceDetector:

	"""
	FaceDetector is a wrapper for the cascade classifiers.
	Must be initialized using faceClassifierPath and eyeClassifierPath, and 
	should only be initialized once per program instance. The only "public"
	method is detect().
	"""

	def __init__(self, faceClassifierPath, eyeClassifierPath):
		"""
		Initialize & Load Haar Cascade Classifiers.
		
		Args:
			faceClassifierPath (string): path to face Haar classifier
			eyeClassifierPath (string): path to eye Haar classifier
		"""
		self.faceClassifier = cv2.CascadeClassifier(faceClassifierPath);
		self.eyeClassifier = cv2.CascadeClassifier(eyeClassifierPath);
	
	def detect(self,frames, faceRect=False):
		"""
		Detect face and eyes. 
		Runs Haar cascade classifiers. Sometimes it is desirable to speed up 
		processing by using a previously-found face rectangle. To do this, pass 
		the old faceRect as the second argument.
		
		Args:
			frames (dict of numpy array): dictionary containing images with different scales
			faceRect (numpy array): array of face rectangle. Face detected if 
									 omitted.
		Returns:
			a dictionary with three elements each representing a rectangle
		"""

		faceIMG = frames['worldBW'];
		faceRects = self.classifyFace(faceIMG);
			
		return faceRects;

		# # Data structure to hold frame info
		# rects = {
		# 	'face': np.array([],dtype=np.int32)
		# };
		
		# # Detect face if old faceRect not provided
		# if faceRect is False or len(faceRect) is 0:
		# 	faceIMG = frames['worldBW'];
		# 	faceRects = self.classifyFace(faceIMG);
			
		# 	return faceRects;

		# 	# Ensure a single face found
		# 	if len(faceRects) is 1:
		# 		faceRect = faceRects[0];
		# 	else:
		# 		# TODO throw error message
		# 		print "No Faces / Multiple Faces Found!";
		# 		return rects;
			
		# rects['face'] = faceRect;


#		# EYE DETECTION 
#		# [not currently used]
#
#
# 		# Extract face coordinates, calculate center and diameter
# 		x1,y1,x2,y2 = rects['face'];
# 		faceCenter = (((x1+x2)/2.0), ((y1+y2)/2.0));
# 		faceDiameter = y2-y1;
		
# 		# Extract eyes region of interest (ROI), cropping mouth and hair
# 		eyeBBox = np.array([x1,
# 		                      (y1 + (faceDiameter*0.24)),
# 		                      x2,
# 		                      (y2 - (faceDiameter*0.40))],dtype=np.int32);
		
		                    
# #		eyesY1 = (y1 + (faceDiameter * 0.16));
# #		eyesY2 = (y2 - (faceDiameter * 0.32));
# #		eyesX1 = x1 * EYE_SCALE;
# #		eyesX2 = x2 * EYE_SCALE;
# #		eyesROI = img[eyesY1:eyesY2, x1:x2];

# 		# Search for eyes in ROI
# 		eyeRects = self.classifyEyes(frames['eyes'],eyeBBox);
# #		print eyeRects;
		
# 		# Ensure (at most) two eyes found
# 		if len(eyeRects) > 2:
# 			# TODO throw error message (and perhaps return?)
# 			print "Multiple Eyes Found!";
# 			# TODO get rid of extras by either:
# 			#	a) using two largest rects or
# 			#	b) finding two closest matches to average eyes
			

# 		# Loop over each eye
# 		for e in eyeRects:
# 			# Adjust coordinates to be in faceRect's coordinate space
# #			e += np.array([eyesX1, eyesY1, eyesX1, eyesY1],dtype=np.int32);
						
# 			# Split left and right eyes. Compare eye and face midpoints.
# 			eyeMidpointX = (e[0]+e[2])/2.0;
# 			if eyeMidpointX < faceCenter[0]:
# 				rects['eyeLeft'] = e; # TODO prevent overwriting
# 			else:
# 				rects['eyeRight'] = e;
# 		# TODO error checking
# 		# TODO calculate signal quality
# 		print 'final rects=',rects
		
		#return rects;

	def classify(self, img, cascade, minSizeX=40):
		"""Run Cascade Classifier on Image"""
		minSizeX = int(round(minSizeX));
#		print 'minSizeX:',minSizeX
		# Run Cascade Classifier
		rects = cascade.detectMultiScale(
				img, minSize=(minSizeX,minSizeX), 
				flags=cv.CV_HAAR_SCALE_IMAGE);
		
		# No Results
		if len(rects) == 0:
			return np.array([],dtype=np.int32);
		
		rects[:,2:] += rects[:,:2]; # ? ? ? 
		rects = np.array(rects,dtype=np.int32);
		return rects;
	
	def classifyFace(self,img):
		"""Run Face Cascade Classifier on Image"""
		rects = self.classify(img,self.faceClassifier,100);#,img.shape[1]*FACE_MIN_SIZE);
		return rects;
		#return rects/FACE_SCALE;
	
	def classifyEyes(self,img,bBox):
		"""Run Eyes Cascade Classifier on Image"""
		EYE_MIN_SIZE = 0.15;
		bBoxScaled = bBox*EYE_SCALE;
		eyesROI = img[bBoxScaled[1]:bBoxScaled[3], bBoxScaled[0]:bBoxScaled[2]];
		
		eyesROI = cv2.equalizeHist(eyesROI);
		
#		print 'eyesROI dimensions: ',eyesROI.shape;
		minEyeSize = eyesROI.shape[1]*EYE_MIN_SIZE;
#		print 'minEyeSize:',minEyeSize;
		cv2.imshow("eyesROI",eyesROI);
		rectsScaled = self.classify(eyesROI, self.eyeClassifier, 
									minEyeSize);
		
#		print rectsScaled;
		# Scale back to full size
		rects = rectsScaled / EYE_SCALE;
		
		# Loop over each eye
		for eye in rects:
			# Adjust coordinates to be in faceRect's coordinate space
			eye += np.array([bBox[0],bBox[1],bBox[0],bBox[1]]);

		return rects;

class FaceModel:

	"""
	FaceModel integrates data from the new frame into a model that keeps track of where the eyes are. To do this it uses:
		- A moving average of the most recent frames
		- Facial geometry to fill in missing data
	The resulting model generates a set of two specific regions of interest (ROI's) where blinking is expected to take place.
	"""
	
	# TODO flush eye history whenever faceRect midpoint changes
	# TODO flush eye history whenever eye rectangle outside of faceRect bbox
	# TODO make sure that eye rectangles don't overlap

	QUEUE_MAXLEN = 50;
	
	QUALITY_QUEUE_MAXLEN = 30;
	qualityHistory = {
		'face':deque(maxlen=QUALITY_QUEUE_MAXLEN)
	};
	
	# Queues storing most recent position rectangles, used to calculate
	# moving averages
	rectHistory = {
		'face': deque(maxlen=QUEUE_MAXLEN)
	};
	
	# Moving average of position rectangles
	rectAverage = {
		'face': np.array([])
	};
	
	def add(self,rects):
		"""Add new set of rectangles to model"""
		
		# Checks to see if face has moved significantly. If so, resets history.
		if(self._faceHasMoved(rects['face'])):
			self.clear();
				
		# Loop over rectangles, adding non-empty ones to history
		for key,rect in rects.items():
			if len(rect) is not 4:
				self.qualityHistory[key].append(0);
				continue;
			self.rectHistory[key].append(rect);
			self.qualityHistory[key].append(1);
#			print 'appended to qHist[',key,']';
		
		# Update moving average stats
		self._updateAverages();

	def getPreviousFaceRects(self):
		if len(self.rectHistory['face']) is 0:
			return np.array([],dtype=np.int32);
		else:
			return self.rectHistory['face'][-1];
	
	
	def getFaceRect(self):
		"""Get face rectangle"""
		return self.rectAverage['face'];
		
	def clear(self):
		""" Resets Eye History"""
		for key,value in self.rectAverage.items():
			self.rectAverage[key] = np.array([],dtype=np.int32);
			self.rectHistory[key].clear();
			self.qualityHistory[key].clear();

	def _faceHasMoved(self, recentFaceRect):
		"""Determines if face has just moved, requiring history reset"""
	
		# If no face found, return true
		if(len(recentFaceRect) is not 4):
			return True;

		history = self.rectHistory['face'];
		
		if len(history) is not self.QUEUE_MAXLEN:
			return False;

		old = history[self.QUEUE_MAXLEN - 10];
		oldX = (old[0] + old[2]) / 2.0;
		oldY = (old[1] + old[3]) / 2.0;
		recentX = (recentFaceRect[0] + recentFaceRect[2]) / 2.0;
		recentY = (recentFaceRect[1] + recentFaceRect[3]) / 2.0;
		change = ((recentX-oldX)**2 + (recentY-oldY)**2)**0.5; # sqrt(a^2+b^2)
		return True if change > 15 else False;

	def _updateAverages(self):
		"""Update position rectangle moving averages"""
		for key,queue in self.rectHistory.items():
			if len(queue) is 0:
				continue;
			self.rectAverage[key] = sum(queue) / len(queue);
		
		faceQ = np.mean(self.qualityHistory['face']);
		
#		print 'Quality:    ', faceQ, eyeLeftQ, eyeRightQ;
#		print 'QHistory: ', self.qualityHistory['face'], self.qualityHistory['eyeLeft'], self.qualityHistory['eyeRight'];
#		print '--------------';

		#print 'QHistSizes: ', len(self.qualityHistory['face']), len(self.qualityHistory['eyeLeft']), len(self.qualityHistory['eyeRight']);

class Util:

	@staticmethod
	def contrast(img, amount='auto'):
		"""
		Modify image contrast
		
		Args:
			img (np array)			Input image array
			amount (float or string)  	Either number (e.g. 1.3) or 'auto'
		"""
		
		pilIMG = Image.fromarray(img);
		
		if amount is 'auto':
			pilEnhancedIMG = ImageOps.autocontrast(pilIMG, cutoff = 0);
			return np.asarray(pilEnhancedIMG);
		else:
			pilContrast = ImageEnhance.Contrast(pilIMG);
			pilContrasted = pilContrast.enhance(amount);
			return np.asarray(pilContrasted);

	@staticmethod
	def threshold(img, thresh):
		"""Threshold an image"""
		
		pilIMG1 = Image.fromarray(img);
		pilInverted1 = ImageOps.invert(pilIMG1);
		inverted = np.asarray(pilInverted1);
		r, t = cv2.threshold(inverted, thresh, 0, type=cv.CV_THRESH_TOZERO);
		pilIMG2 = Image.fromarray(t);
		pilInverted2 = ImageOps.invert(pilIMG2);
		thresholded = np.asarray(pilInverted2);
		return thresholded;

	
	@staticmethod
	def equalizeHSV(img, equalizeH=False, equalizeS=False, equalizeV=True):
		"""
		Equalize histogram of color image using BSG2HSV conversion
		By default only equalizes the value channel
		
		Note: OpenCV's HSV implementation doesn't capture all hue info, see:
		http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CvtColor
		http://www.shervinemami.info/colorConversion.html
		"""

		imgHSV = cv2.cvtColor(img,cv.CV_BGR2HSV);
		h,s,v = cv2.split(imgHSV);
		
		if equalizeH:
			h = cv2.equalizeHist(h);
		if equalizeS:
			s = cv2.equalizeHist(s);
		if equalizeV:
			v = cv2.equalizeHist(v);
		
		hsv = cv2.merge([h,s,v]);
		bgr = cv2.cvtColor(hsv,cv.CV_HSV2BGR);
		return bgr;

class PersonModel:

	@staticmethod
	def isFaceGazing(gaze,faceRects):
		if len(faceRects) is 0:
			return False;

		distances = np.zeros((len(faceRects),1));
		i = 0;
		for face in faceRects:
			x1,y1,x2,y2 = face;
			faceCenter = (((x1+x2)/2.0), ((y1+y2)/2.0));
			distances[i] = math.sqrt((faceCenter[0] - gaze[0])**2 + (faceCenter[1] - gaze[1])**2)
			i = i + 1;
		minDistance = np.amin(distances);

		print distances.flatten();

		return minDistance < 100;

	@staticmethod
	def getBodyRects(faceRects):

		if len(faceRects) == 0:
			return [[]]

		bodyRects = np.zeros((faceRects.shape[0],4));

		i = 0;
		for face in faceRects:

			 # (x1,y1) is top left (TL) corner
			 # (x2,y2) is bottom right (BR) corner
			x1, y1, x2, y2 = face.astype(np.int32);
			
			# rect is square, so width and height equivalent
			size = x2 - x1;

			bodyWidthFactor = 2;
			bodyHeightFactor = 3;

			beyondFaceX = ((size*bodyWidthFactor)-size)/2.0;

			bodyX1 = x1 - beyondFaceX;
			bodyY1 = y2;
			bodyX2 = x2 + beyondFaceX;
			bodyY2 = y2 + (size*bodyHeightFactor);

			bodyRects[i]=np.asarray([bodyX1, bodyY1, bodyX2, bodyY2],dtype=np.int32);
			i = i + 1;

		return bodyRects;

			
class Display:

	def renderScene(self, frame, model, rects=False):
		"""Draw face and eyes onto image, then display it"""
		
		# Get Coordinates
		faceRect = model.getFaceRect();
		linePoints = model.getEyeLine();
	
		# Draw Shapes and display frame
		self.drawLine(frame, linePoints[0],linePoints[1],(0, 0, 255));
		self.drawRectangle(frame, faceRect, (0, 0, 255));
		
		cv2.imshow("Video", frame);
	
	@staticmethod
	def drawHistogram(img,color=True,windowName='drawHistogram'):
		h = np.zeros((300,256,3))
		 
		bins = np.arange(256).reshape(256,1)
		
		if color:
			channels =[ (255,0,0),(0,255,0),(0,0,255) ];
		else:
			channels = [(255,255,255)];
		
		for ch, col in enumerate(channels):
			hist_item = cv2.calcHist([img],[ch],None,[256],[0,255])
			#cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
			hist=np.int32(np.around(hist_item))
			pts = np.column_stack((bins,hist))
			#if ch is 0:
			cv2.polylines(h,[pts],False,col)
		 
		h=np.flipud(h)
		 
		cv2.imshow(windowName,h);
	
	@staticmethod
	def drawLine(img, p1, p2, color):
		"""Draw lines on image"""
		p1 = (int(p1[0]*DISPLAY_SCALE), int(p1[1]*DISPLAY_SCALE));
		p2 = (int(p2[0]*DISPLAY_SCALE), int(p2[1]*DISPLAY_SCALE));
		cv2.line(img, p1, p2,(0, 0, 255));
	
	@staticmethod
	def drawRectangle(img, rect, color):
		"""Draw rectangles on image"""
		
		if len(rect) is not 4:
			# TODO throw error
			return;

		# if points out of bounds, adjust to prevent gruesome errors
		# remove negative values, replace with zeros
		rect = rect.clip(0); 
		
		# Trim points beyond max width
		if rect[2] >= img.shape[1]:
			rect[2]=img.shape[1]-1;

		# Trim points beyond max height
		if rect[3] >= img.shape[0]:
			rect[3]=img.shape[0]-1;

		# rect = rect * DISPLAY_SCALE;
		x1, y1, x2, y2 = rect.astype(np.int32);
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2);

	@staticmethod
	def drawBodies(img,faceRects,bodyRects):
		"""Draw body and face rectangles on image"""

		for face in faceRects:
			disp.drawRectangle(img, face, (0, 0, 255));

		for body in bodyRects:
			disp.drawRectangle(img,body, (0, 255, 0));


class Capture:

	height = 0;
	width = 0;

	EYE_SCALE = 0.5;
	WORLD_SCALE = 0.5;

	PUPIL_XMIN = 0;
	PUPIL_XMAX = 400;
	PUPIL_YMIN = 130;
	PUPIL_YMAX = 350;
	
	def __init__(self, cameraEye, cameraWorld):

		self.cameraEye = cameraEye;
		self.cameraWorld = cameraWorld;
	
		# # Setup webcam dimensions
		# self.height = self.camera.get(cv.CV_CAP_PROP_FRAME_HEIGHT);
		# self.width = self.camera.get(cv.CV_CAP_PROP_FRAME_WIDTH);
		
		# # Reduce Video Size to make Processing Faster
		# if scaleFactor is not 1:
		# 	scaledHeight = self.height / scaleFactor;
		# 	scaledWidth = self.width / scaleFactor;
		# 	self.camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT,scaledHeight);
		# 	self.camera.set(cv.CV_CAP_PROP_FRAME_WIDTH,scaledWidth);
	
		# # Create window
		# cv2.namedWindow("Video"+str(cameraIndex), cv2.CV_WINDOW_AUTOSIZE);
	
	def read(self):

		#######################################################
		# CAPTURE, SCALE & RECOLOR WORLD IMAGE							
		#######################################################
		frameWorldRetVal, frameWorld = self.cameraWorld.read();
		scaledWorldColor = cv2.resize(frameWorld,None,fx=self.WORLD_SCALE,fy=self.WORLD_SCALE);
		scaledWorldBW = cv2.cvtColor(scaledWorldColor,cv.CV_BGR2GRAY);

		#######################################################
		# CAPTURE, SCALE & CROP EYE IMAGE							
		#######################################################
		frameEyeRetVal, frameEye = self.cameraEye.read();
		# eye camera returns frames that are 800px high by 1280px wide
		scaledEye = cv2.resize(frameEye,None,fx=self.EYE_SCALE,fy=self.EYE_SCALE);
		scaledEye = cv2.cvtColor(scaledEye,cv.CV_BGR2GRAY);
		# remove pixels not centered on eye 
		scaledEye = scaledEye[self.PUPIL_YMIN:self.PUPIL_YMAX, self.PUPIL_XMIN:self.PUPIL_XMAX];
		
		frames = {
			'worldColor': scaledWorldColor,
			'worldBW': scaledWorldBW,
			'eye': scaledEye
		};
		
		return frames;

def main():
	# Instantiate Classes
	# detector = FaceDetector(FACE_CLASSIFIER_PATH, EYE_CLASSIFIER_PATH);
	#model = FaceModel();
	#display = Display();
	captureEye = Capture(2);
	captureWorld = Capture(3);
	
	oldTime = time.time();
	i = 0;
	
	while True:
		# Calculate time difference (dt), update oldTime variable
		newTime = time.time();
		dt =  newTime - oldTime;
		oldTime = newTime;
		
		# Grab Frames
		framesEye = captureEye.read();
		framesWorld = captureWorld.read();
		
		# # Detect face 20% of the time, eyes 100% of the time
		# if i % 5 is 0:
		# 	rects = detector.detect(frames);
		# else:
		# 	rects = detector.detect(frames,model.getPreviousFaceRects());
		# i += 1;
	
		# # Add detected rectangles to model
		# model.add(rects);
		
		# # Render
		cv2.imshow("Video2", framesEye['display']);#displayFrame);
		cv2.imshow("Video3", framesWorld['display']);
		# display.renderScene(frames['display'],model,rects);
		# display.renderEyes(frames['color'],model);

def fit_polynomial_surf(X,Y,Z):         
    """
    Takes three lists of points and     
    performs Singular Value Decomposition
    to find a linear least squares fit surface
    
    This code was taken from the PUPIL project, which can be retrieved with:
    git clone https://code.google.com/p/pupil/
    """

    One = np.ones(Z.shape)
    Zero = np.zeros(Z.shape)
    XX = X*X
    YY = Y*Y
    XY = X*Y
    XXYY = X*Y*X*Y
    V = np.vstack((One,X,Y,XX,YY,XY,XXYY))   
    V = V.transpose()
    U,w,Vt = np.linalg.svd(V,full_matrices=0);
    V = Vt.transpose();
    Ut = U.transpose();
    pseudINV = np.dot(V, np.dot(np.diag(1/w), Ut));
    coefs = np.dot(pseudINV, Z);
    c, x,y,xx,yy,xy,xxyy = coefs
    """
    print "coeffs"
    print "x",x
    print "y",y
    print "xy",xy
    print "xx",xx
    print "yy",yy
    print "xxyy",xxyy
    print "c",c
    """
    return x,y,xx,yy,xy,xxyy,c

def calibrate(eyepts, worldpts):
	"""
	take a list of eye coords [[x,y], [x,y]...] and world coords and
	fit a 2nd degree polynomial surface to them
	"""
	worldx = np.array(worldpts)[:,0];	
	worldy = np.array(worldpts)[:,1];	
	eyex = np.array(eyepts)[:,0];	
	eyey = np.array(eyepts)[:,1];	

	xcoeff = fit_polynomial_surf(eyex, eyey, worldx);
	ycoeff = fit_polynomial_surf(eyex, eyey, worldy);

	print "xcoeff";	
	print xcoeff;	
	print "ycoeff";	
	print ycoeff;	
	return xcoeff, ycoeff;
 
def getWorldCoords((eyex, eyey), ax, ay):
	"""
	Returns the world position x,y from the given eye coordinates
	using the coefficients matrix a with coefficients for terms
	x,y,xx,yy,xy,xxyy,c listed in that order
	"""

	x = ax[6] + ax[5]*eyex*eyex*eyey*eyey + ax[4]*eyex*eyey + ax[3]*eyey*eyey + ax[2]*eyex*eyex + \
		ax[1]*eyey + ax[0]*eyex;
	y = ay[6] + ay[5]*eyex*eyex*eyey*eyey + ay[4]*eyex*eyey + ay[3]*eyey*eyey + ay[2]*eyex*eyex + \
		ay[1]*eyey + ay[0]*eyex;
	return [x,y];


 
# THESE INDICES CAN CHANGE AT A MOMENTS NOTICE
# use a for loop to check for correct camera
dev1 = "0";
dev2 = "1";

# SKIN COLOR CODE
# create skin color histogram in YCrCb space
#skinhist = numpy.zeros((256,256));
#skinhist = skinhist.astype(np.uint8);
##cv2.ellipse(skinhist, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])None
#cv2.ellipse(skinhist, (113, 155.6), (23.4, 15.2), 43.0, 0.0, 360.0, (255, 255, 255);
#cv2.imshow("skinhist",skinhist);

#http://spottrlabs.blogspot.com/2012/01/super-simple-skin-detector-in-opencv.html
#bool isSkin(const Scalar& color) { 
#    Mat input = Mat(Size(1, 1), CV_8UC3, color);
#    Mat output;
#
#    cvtColor(input, output, CV_BGR2YCrCb);
#
#    Vec3b ycrcb = output.at<Vec3b>(0, 0);
#    return ((skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0));
#}
#

 
# get user to define correct camera devices for eye and world cameras
while dev1 != "":
  d1 = int(dev1);
  cameraEye = cv2.VideoCapture(d1);
  if not cameraEye.isOpened():
    die("eye camera failed on device " + dev1);
  frameEyeRetVal, frameEye = cameraEye.read();
  scaledEye = cv2.resize(frameEye,None,fx=0.5,fy=0.5);
  cv2.imshow('Eye Camera',scaledEye);
  dev1 = raw_input("Provide a new device number for the eye camera " + 
	"or hit enter if the correct device has already been selected\n");

dev2 = str((d1+1) % 2);

while dev2 != "":
  d2 = int(dev2);
  cameraWorld = cv2.VideoCapture(int(d2));
  if not cameraWorld.isOpened():
    die("world camera failed on device " + dev2);
  frameWorldRetVal, frameWorld = cameraWorld.read();
  scaledWorld = cv2.resize(frameWorld,None,fx=0.5,fy=0.5);
  cv2.imshow('World Camera',scaledWorld);
  dev2 = raw_input("Provide a new device number for the world camera" + 
	"or hit enter if the correct device has already been selected\n");

cv2.destroyWindow('Eye Camera');
cv2.destroyWindow('World Camera');

disp = Display();
detector = FaceDetector(FACE_CLASSIFIER_PATH, EYE_CLASSIFIER_PATH);
capture = Capture(cameraEye,cameraWorld);

  
# [used in gray projection]
horizontal_sum = 0;
vertical_sum = 0;

i = 0;

# make a dynamically cropped frame
crop = [0,0,frameEye.shape[1], frameEye.shape[0]];

eyepts_initialized = False;
world_initialized = False;
done = False;
done_accum = False;
eyepts = None;

while True:
	i = i+1;

	frames = capture.read();

	faceRects = detector.detect(frames);
	bodyRects = PersonModel.getBodyRects(faceRects);

	Display.drawBodies(frames['worldColor'],faceRects,bodyRects);

	#######################################################
	# THRESHOLDING #
	#######################################################

	# Blur image, eliminating high spatial frequency dark spots
	blurred1 = cv2.GaussianBlur(frames['eye'],(21,21),1);

	# Dynamic threshold: slightly higher than lowest light intensity
	minval = np.min(blurred1);
	threshval = minval + 25;

	# Run threshold, replace gray with white
	threshed_retval,threshed = cv2.threshold(blurred1, threshval, 
		maxval=255, type=cv.CV_THRESH_BINARY);
	matr = np.where(threshed == threshval);
	threshed[matr] = 255;

	# Blur, then apply canny edge detection
	blurred = cv2.GaussianBlur(threshed,(7,7),1);
	edges = cv2.Canny(blurred,15,30);
	
	edgePoints = np.argwhere(edges>0);
	gotedgePoints = False;

	# needs to be more robust
	# if the edges were defined in the canny image try to find the pupil
	if edgePoints.shape[0] > 6:
		gotedgePoints = True;
		ellipseBox = cv2.fitEllipse(edgePoints);
		eBox = tuple([tuple([ellipseBox[0][1],ellipseBox[0][0]]),\
		tuple([ellipseBox[1][1],ellipseBox[1][0]]),ellipseBox[2]*-1]);
		
		ellipseFrame = frames['eye'].copy();
		ellipseFrame = cv2.cvtColor(ellipseFrame,cv.CV_GRAY2BGR);
		cv2.ellipse(ellipseFrame,eBox,(0, 255, 0));
		
		# the center of the elipse is our pupil estimate
		center = np.asarray(eBox[0], dtype=np.int32);
		center = np.around(center);
		center = tuple(center);
		cv2.circle(ellipseFrame, center, 3,(255,0,0));
		cv2.imshow("ellipseFit",ellipseFrame);

	# print a message to indicate calibration is about to start
	if i == 10:
		print "please focus on the red cross in the image."
		print "now turn your head until you are dizzy";

	#######################################################
	# CALIBRATION						
	#######################################################

	# assume by i == 20 the user is looking at the gray square
	if i > 20 and not done:

		# Search for target
		targetFound, worldcenters = \
		cv2.findCirclesGridDefault(frames['worldBW'], (4,11), \
		flags=cv2.CALIB_CB_ASYMMETRIC_GRID); 

		# Calibration target found
		if targetFound:
			worldpt = worldcenters.sum(0)/worldcenters.shape[0];
			cv2.circle(frames['worldBW'], tuple(worldpt[0]), \
				3, (255, 100, 255));
			
			# if the pupil center was calculated, try to store a point
			if edgePoints.shape[0] > 6:
				# find the calibration image 
				if eyepts_initialized == True:
					print str(len(eyepts));
					cv2.circle(frames['worldColor'], tuple([worldpt[0][0],worldpt[0][1]]),5, (0, 0, 255));
					worldpts.append([worldpt[0][0], \
							 worldpt[0][1]]);
					eyepts.append([center[0],center[1]]);
				else:
					eyepts = [[center[0],center[1]]];
					worldpts = [[worldpt[0][0], \
							 worldpt[0][1]]];
					eyepts_initialized = True;

			# If enough data collected, run calibration routine
			if eyepts is not None and len(eyepts) > 200:
				done = True
				print "worldpts";
				print worldpts;
				print "eyepts";	
				print eyepts;	
				xcoeff, ycoeff = calibrate(eyepts, worldpts);
		
		# Calibration target not found
		else:
			print '...';

	#######################################################
	# MAIN LOOP, POST-CALIBRATION					
	#######################################################
	if done == True:
		gazept = getWorldCoords(center, xcoeff, ycoeff);
		gazept = np.around(np.asarray(gazept, dtype=np.int32));
		# print gazept;
		print PersonModel.isFaceGazing(gazept,faceRects);
		cv2.circle(frames['worldColor'], tuple(gazept),GAZE_RADIUS*2, (255, 100, 255), 5);
	
	cv2.imshow('TheWorld',frames['worldColor']);

	#######################################################
	# DIAGNOSTIC DISPLAYS	
	# Uncomment as needed.							
	#######################################################

	# Display Blurred Image Used in Thresholding
	#cv2.imshow("Blurred",blurred1);

	# Display Histogram to Diagnose Skewed Distribution
	#disp.drawHistogram(blurred1, False);

	# Display Thresholded Image
	#cv2.imshow('Thresholded',threshed);

	# Display Edges
	#cv2.imshow("CannyEdgeDetector",edges);

	# Display Hough Circles
	#cv2.imshow("HoughCircles",houghCircleFrame);

#cProfile.run('main()','profile.o','cumtime');
