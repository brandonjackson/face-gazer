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
	PUPIL_XMAX = 450;
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

def die(msg):
	# print error message and exit
	print "gaze-tracker: " + msg;
	exit(1);

def estimate_pupil(img, minr, maxr):
	# estimates the x,y position of the pupil in img using an integral im
	# REALLY REALLY SLOW RIGHT NOW
	
	# create an integral image ii to do fast pupil location estimation
	ii = cv2.integral(img);

	h = img.shape[0];
	print "h is " + str(h);
	w = img.shape[1];
	print "w is " + str(w);

	print "wrange is " + str(w-6*minr);
	print "hrange is " + str(h-6*minr);
	best = 0;
	bestx = 0;
	besty = 0;	
	# loop through all possible pupil radii r and all positions x,y
	for r in range(minr, maxr+1):
		#print r;
		for y in range(h-6*r-1):
			for x in range(w-6*r-1):
				#print "(x,y) is " + str(x) + " " + str(y);
				bright = ii[y+6*r,x+6*r] - ii[y+6*r,x] - \
					ii[y,x+6*r] - ii[y,x];
				i = x+2*r;
				j = y+2*r;
				dark = ii[j+r,i+r] - ii[j+r,i] - \
					ii[j,i+r] - ii[j,i];
				matchval = bright - 2*dark;
				if matchval > best:
					best = matchval;
					bestx = i + r;
					besty = j + r;

	return x, y;

def est_pupil_template(img, minr, maxr):
	# estimates the x,y position of the pupil in img using template matchng 
	# currently doesn't pick out the pupil 
	
	h = img.shape[0];
	w = img.shape[1];

	best = w*h;
	# loop through all possible pupil radii r and all positions x,y
	for r in range(minr, maxr+1):
		templ = np.ones((4*r,4*r))*255;
		templ[r:3*r,r:3*r] = 0;
		templ = templ.astype(np.uint8);
		
		cv2.imshow('template', templ);
		result = cv2.matchTemplate(img, templ,cv.CV_TM_SQDIFF_NORMED); 
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result);
		
		print minVal;
		print maxVal;
		if minVal < best:
			best = minVal;
			bestloc = [minLoc[0],minLoc[1]];
			#bestloc[0] = bestloc[0] + 6*r;
			#bestloc[1] = bestloc[1] + 6*r;
				
	return tuple(bestloc);

	
def bestcircle(img, circles, rmin, rmax):
	# estimates the x,y position of the pupil in img using an integral
	# image and checking only points indicated by the Hough circles array 

	temp = img/255;

	# create an integral image ii to do fast pupil location estimation
	ii = cv2.integral(temp);

	h = img.shape[0];
	w = img.shape[1];

	best = -w*h;
	bestx = 0;
	besty = 0;	

	for c in circles[0]:
		for r in range(rmin, rmax):
			# define scale of bright box radius bbr
			bbr = 3;
		 
			xcent = c[0];
			ycent = c[1];
		
			# assign bright box corners bbr*r away from center
			x1 = max(0,xcent-bbr*r);
			y1 = max(0,ycent-bbr*r);
			x2 = min(w,xcent+bbr*r);
			y2 = min(h,ycent+bbr*r);
			area = (y2-y1)*(x2-x1);	
			bright = ii[y2,x2] - ii[y2,x1] - ii[y1,x2] + ii[y1,x1];
		
			# assign dark box corners r away from center
			x1 = max(0,xcent-r);
			y1 = max(0,ycent-r);
			x2 = min(w,xcent+r);
			y2 = min(h,ycent+r);
			dark = ii[y2,x2] - ii[y2,x1] - ii[y1,x2] + ii[y1,x1];
		
			matchval = (bright - 2*dark)/area;
			if matchval > best:
				best = matchval;
				bestc = c;
				bestr = r;
	
	return bestc, bestr;

def findcalim(img, templ):
	# returns the best location of the template
	result = cv2.matchTemplate(img, templ,cv.CV_TM_SQDIFF_NORMED); 
	cv2.imshow('result', result);
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result);
	return maxLoc;	


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
eyePoints_initialized = False;
eyedata_initialized = False;
done_accum = False;
eyepts = None;

while True:
	i = i+1;

	frames = capture.read();

	faceRects = detector.detect(frames);
	bodyRects = PersonModel.getBodyRects(faceRects);

	Display.drawBodies(frames['worldColor'],faceRects,bodyRects);

	#for face in faceRects:
		#print face;
		#disp.drawRectangle(frames['worldColor'], face, (0, 0, 255));
	
	# experimenting with RGB image in order to crop eye more dynamically
	# ideas: use blue image to find glasses to crop image quickly
	# then use blue image to find pupil and crop box around it
	# B, G, R = cv2.split(frames['eye']);
	# cv2.imshow('BEye', B);
	# cv2.imshow('GEye', G);
	# cv2.imshow('REye', R);

	# # Good Features To Track
	# corners = cv2.goodFeaturesToTrack(frames['eye'], 50, .1, 20);
	# featureFrame = cv2.cvtColor(frames['eye'],cv.CV_GRAY2BGR);
	# if corners is not None:
	# 	corners = np.reshape(corners, (-1,2));
	# 	for x, y in corners:
	# 		cv2.circle(featureFrame,(x,y),3,(0, 255, 0));

	
	if eyedata_initialized == False:
		eyellipse = np.ones(frames['eye'].shape)*255;
		eyellipse = eyellipse.astype(np.uint8);
		eyedata_initialized = True;

	#######################################################
	# THRESHOLDING 										  #
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

	# use on center - off surround template to quickly estimate pupil
	# center over all x,y and pupil radius ranging from minr to maxr 
	minr = int(round(threshed.shape[1]*0.03));
	maxr = int(round(threshed.shape[1]*0.1));
	
	
	#pupx, pupy = estimate_pupil(threshed, minr, minr);
		
	# code to check size of minr and maxr
	# display minr and maxr in a separate window
	# print minr;
	# print maxr;
	# lineFrame = frames['eye'].copy();
	# lineFrame = cv2.cvtColor(lineFrame,cv.CV_GRAY2BGR);
	# cv2.line(lineFrame,tuple([200,50]), tuple([200+minr, 50]), (200, 100, 255));
	# cv2.line(lineFrame,tuple([200,75]), tuple([200+maxr, 75]), (100, 255, 0));
	# cv2.imshow("lines",lineFrame);
	
	# TODO update this based on previously-found iris radii
	minDist = maxr*2;
	circles = cv2.HoughCircles(threshed, cv.CV_HOUGH_GRADIENT, 2, \
		minDist, param1=30, param2=10,minRadius=minr,maxRadius=maxr);
	
	houghCircleFrame = cv2.cvtColor(frames['eye'],cv.CV_GRAY2BGR);
	
	# # Hough circle plotting
	# if circles is not None and len(circles)>0:
	# 	for c in circles[0]:
	# 		c = c.astype(np.int32);
	# 		cv2.circle(houghCircleFrame,(c[0],c[1]),c[2],(0, 255, 0));

	if circles is not None and len(circles)>0:
		c, r = bestcircle(threshed, circles, minr, maxr);
		c = c.astype(np.int32);
		pupROI = [c[0]-2*r,c[1]-2*r, c[0]+2*r,c[1]+2*r];
		cv2.circle(houghCircleFrame, (c[0]+2,c[1]+2), c[2], (255, 100, 255));
		cv2.rectangle(houghCircleFrame, (c[0]-2*r,c[1]-2*r), (c[0]+2*r,c[1]+2*r), (255, 100, 255));

		blurredROI = cv2.GaussianBlur(threshed, (15,15) ,1);#[pupROI[1]:pupROI[3], pupROI[0]:pupROI[2]], (7,7) ,1);
		edgesROI = cv2.Canny(blurredROI,15,30);
		conROI = edgesROI;
		contours, hierarchy = cv2.findContours(edgesROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);
		best = 0;
		for cnt in contours:
			retval = cv2.contourArea(cnt);
			if retval > best:
				bestcnt = cnt;	
		edgesROI = cv2.cvtColor(edgesROI,cv.CV_GRAY2BGR);
#		cv2.drawContours(edgesROI,bestcnt,-1,(255,255,0),-1);
#		cv2.imshow("contours",edgesROI);

	# estimate pupil using template matching
	#loc = est_pupil_template(frames['eye'], maxr, maxr);
	#cv2.circle(houghCircleFrame, loc, 10, (50, 255, 100));
	

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

	# use data accumulated during calibration constrain ROI
	# and radius for pupil
	# if i >= 20 and done_accum is False:
	# 	if gotedgePoints == True:
	# 		# add the center to eyePoints array and display
	# 		# accumulated points
	# 		if eyePoints_initialized is True:
	# 			eyePoints = np.vstack(\
	# 				[eyePoints, [center[1],center[0]]]);
	# 			eyellipse[center[1],center[0]] = 0;
	# 			cv2.imshow("eyellipse",eyellipse);
	# 		else:
	# 			eyePoints = np.array(\
	# 				[[center[1],center[0]]]);
	# 			eyePoints_initialized = True;	
	
	# 	# if there are enough accumulated points to fit an ellipse
	# 	# fit an ellipse to the points
	# 	if eyePoints_initialized is True and eyePoints.shape[0] > 6:
	# 		ellipseBox = cv2.fitEllipse(eyePoints);
	# 		eBox = tuple([tuple([ellipseBox[0][1],\
	# 		ellipseBox[0][0]]),tuple([ellipseBox[1][1],\
	# 		ellipseBox[1][0]]),ellipseBox[2]*-1]);
	# 		cv2.ellipse(ellipseFrame,eBox,(0, 0, 255));
	# 		cv2.imshow("ellipseFit",ellipseFrame);
			
	# 	# if calibration is done then accumulation is done
	# 	if done:
	# 		done_accum = True;
			
	# 		# show the elipse in a new image frame
	# 		dat = np.ones(frames['eye'].shape);
	# 		dat = dat.astype(np.uint8);
	# 		cv2.ellipse(dat,eBox,0);
	# 		cv2.ellipse(eyellipse,eBox,0);
	# 		cv2.imshow("eyellipse",eyellipse);
			
	# 		# find the points on the elipse and construce a
	# 		# upright bounding box around them
	# 		edgePoints = np.argwhere(dat==0);
	# 		maxvals = np.amax(edgePoints, axis=0);
	# 		minvals = np.amin(edgePoints, axis=0);
	# 		print minvals;
	# 		print maxvals;	
	# 		height = maxvals[1]-minvals[1];
	# 		width = maxvals[0]-minvals[0];
	# 		newheight = height*2;
	# 		newwidth = width*2;
	# 		newminvals = (minvals - minvals/2).astype(int); 
	# 		newmaxvals = minvals + [newheight,newwidth];
			
	# 		print newminvals;
	# 		print newmaxvals;
			
	# 		scaledxmin = newminvals[0]; 
	# 		scaledymin = newminvals[1]; 
	# 		scaledxmax = newmaxvals[0]; 
	# 		scaledymax = newmaxvals[1]; 
	# 		#contours, hierarchy = cv2.findContours(dat,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
	# 		#bBox = cv2.boundingRect(contours[0]);	
	# 		#cv2.rectangle(eyellipse,(minvals[1],minvals[0]),\
	# 		#	(maxvals[1],maxvals[0]),20);
	# 		cv2.imshow("eyellipse",eyellipse);
	

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
			if eyepts is not None and len(eyepts) > 30:
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
	
	# detect skin color
	# skinhist = np.zeros((256,256));
	
	cv2.imshow('TheWorld',frames['worldColor']);

	#######################################################
	# DIAGNOSTIC DISPLAYS	
	# Uncomment as needed.							
	#######################################################

	# Display Blurred Image Used in Thresholding
	# cv2.imshow("Blurred",blurred1);

	# Display Histogram to Diagnose Skewed Distribution
	# disp.drawHistogram(blurred1, False);

	# Display Thresholded Image
	# cv2.imshow('Thresholded',threshed);

	# Display Edges
	# cv2.imshow("CannyEdgeDetector",edges);

	# Display Hough Circles
	# cv2.imshow("HoughCircles",houghCircleFrame);

	# Display GoodFeaturesToTrack
	# cv2.imshow('GoodFeatures', featureFrame);

#cProfile.run('main()','profile.o','cumtime');
