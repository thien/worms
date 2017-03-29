# Durham University

#####################################################################

# Import Libraries

import numpy as np
import os
import cv2
import time
import random
import math
import csv
from scipy import spatial

#####################################################################

# Functions Related to Skeleton Manipulation

def wormDOACheck(data_csv, image, wormblobs, normalised, skeleton, worms_dir, results_img_dir):
	"""
	wormDOACheck

	Checks the contours of each worm blob in order to detect 
	a worms shape and pattern type. A percentage is then returned 
	corresponding to how likely it is that the worm is dead.

	@param data_csv - dictionary corresponding to results
	@param image - thresholded image
	@param wormblobs - contour list of wormblobs
	@param normalised - normalised image
	@param skeleton - skeleton image
	@param worms_dir - directory of individual worms to save in.
	@param results_img_dir - directory for result images to save in.

	"""

	print("--- Initiating Worm DOA Check ---")

	# initiate variables
	death_threshold = 75
	line_percentage_weight = 0.85
	pattern_percentage_weight = 0.15
	column = data_csv['Column']
	worm_number = 0
	worm_statuses = []
	no_worms_found = 0
	no_worms_found_cluster = 0
	no_worm_statuses = 0

	# print # of worms
	print("Number of Worm Blobs Found: " + str(len(wormblobs)))

	result_img = cv2.cvtColor(normalised.copy(),cv2.COLOR_GRAY2RGB)

	# iterate through the worms found
	for worm in wormblobs:
		# increment worm count
		status = ""
		worm_number += 1

		# wormDOAShape(worm_number, worm, image)
		line_percentage = wormDOAShapeSkeleton(worm_number, worm, image, normalised, skeleton)

		if line_percentage != -1:
			no_worms_found += 1
			# image = drawWormNumber(image, worm, worm_number)
			cos_sim = wormDOAPattern(worm,image,normalised)
			# save worm to location

			dead_percentage = round((line_percentage * line_percentage_weight) + (cos_sim * pattern_percentage_weight), 2)

			status_string = "shape: " + str(line_percentage) + ", pattern: " + str(cos_sim) + " = " + str(dead_percentage)

			if dead_percentage >= death_threshold:
				status = "dead"
				no_worm_statuses += 1
				status_string += " (Dead)"
				print("[" + str(worm_number) + "]", "is most likely dead:",status_string)
				cv2.drawContours(result_img,[worm],0,(0,0,255),1)
			else:
				status = "alive"
				print("[" + str(worm_number) + "]",status_string)
				cv2.drawContours(result_img,[worm],0,(0,255,0),1)

			worm_statuses.append(str(worm_number) + " - "+status_string)
		else:
			# this is a cluster of worms.
			status = "cluster"

			# worm_skel_base = np.zeros((skeleton.shape[0], skeleton.shape[1],3), np.uint8)
			worm_skel_img = maskSkeletonWorm(worm, skeleton).astype(np.uint8)
			# worm_skel_img = cv2.add(worm_skel_base,worm_skel_img)
			# issue here.


			intersectionpoints = findIntersections(worm_skel_img);
			endpoints = findEndpoints(worm_skel_img);

			intersection_colour = (255,0,0)
			worm_skel_img = cv2.cvtColor(worm_skel_img,cv2.COLOR_GRAY2RGB)
			for i in intersectionpoints:
				worm_skel_img[i[0]][i[1]] = intersection_colour

			cluster_count = math.floor(len(endpoints)/2)

			print("worms in cluster: ",cluster_count,"?")

			no_worms_found_cluster += cluster_count

			cv2.drawContours(result_img,[worm],0,(0,255,255),1)
			result_img = cv2.add(worm_skel_img,result_img)
			result_img = writeOnWormImage(result_img,worm,str(cluster_count) + "?")
			worm_statuses.append(str(worm_number) + " - "+ "Cluster, May have " + str(cluster_count) + " worms")

		# save individual worm
		saveWormToImage(column, worm_number, worm, image, worms_dir, status)


	print(worm_statuses)
	# save result img into file.
	saveImage(column, result_img, results_img_dir)
	# add wormblobs found to the data
	data_csv['Worm Blobs Found'] = worm_number

	# add actual worm count to data
	data_csv['Worms Found'] = no_worms_found

	# add worm count with cluster check
	data_csv['Worms Found (Clusters)'] = no_worms_found + no_worms_found_cluster

	# add number of dead worms to data
	data_csv['Number of Dead Worms'] = no_worm_statuses

	worm_statuses = '"' + (" \n ".join([ str(v) for v in worm_statuses]))+'"'

	data_csv['Worms Metadata'] = worm_statuses


	# wormcount_percentage = str(int(no_worms_found/data_csv['Ground Truth Worm Count']*100))
	# data_csv['Worm Count Percentage'] = wormcount_percentage

	print("--- Ending Worm DOA Check ---")

def wormDOAShape(i,worm,image):
	"""

	wormDOAShape

	Detects the shape of the worm by comparing the area of the worm
	against its convex hull. This isn't used in the final display.

	@param i - worm number
	@param worm - individual worm contour
	@param image - thresholded image

	"""

	# colour for worm
	colour = 120
	# calculate moment for worm
	M = cv2.moments(worm)

	# calculate area of worm
	area = cv2.contourArea(worm)

	# get the convex hull of each worm
	hull = cv2.convexHull(worm)

	# calculate area difference between the worm and the convex hull of worm
	hull_area = cv2.contourArea(hull)
	worm_area_ratio = round(area/hull_area*100,1)
	# print(area, hull_area, worm_area_ratio)


	if worm_area_ratio > 70:
		# its most likely straight
		print("[" + str(i) + "]! Worm " + "is most likely straight")
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv2.circle(image, (cX,cY),  3, (100,100,100), -1)
	elif worm_area_ratio > 50:
		print("[" + str(i) + "] Worm is most likely curved")
	else:
		print("[" + str(i) + "] Worm is definitely curved")

	# draw the convex hull 
	cv2.drawContours(image,[hull],0,colour,1)

	# draw contour of worms
	cv2.drawContours(image,[worm],0,colour,1)

def maskSkeletonWorm(worm,skeleton):
	"""

	maskSkeletonWorm

	singles out a skeleton from the group.

	@param worm: worm contour
	@param skeleton: skeleton of image
	@return new_img: image of the worm's skeleton

	"""
	# create baseimage for mask
	mask_base = np.zeros(skeleton.shape,np.uint8)
	cv2.drawContours(mask_base,[worm],0,255,-1)

	# apply mask to skeleton so we can isolate the single skeleton.
	new_img = cv2.bitwise_and(skeleton, skeleton, mask=mask_base)
	return new_img

def wormDOAShapeSkeleton(worm_number,worm,image,normalised,skeleton):
	"""

	wormDOAShapeSkeleton

	Detect Shape of Worm by using its skeleton.

	@param worm_number: worm number
	@param worm: worm contour
	@param image: thresholded image
	@param normalised: normalised image
	@param skeleton: skeleton of image

	@return straight_line_percentage: percentage of the worm being straight.

	"""

	straight_line_percentage = 0

	# create baseimage for mask
	# mask_base = np.zeros(image.shape,np.uint8)
	# cv2.drawContours(mask_base,[worm],0,255,-1)

	# apply mask to skeleton so we can isolate the single skeleton.
	individual_worm = maskSkeletonWorm(worm, skeleton)
	# individual_worm = cv2.bitwise_and(skeleton, skeleton, mask=mask_base)

	# showDiagnosticImage("individual_worm", individual_worm)

	# count number of pixels this skeleton contains
	pixels = countSkeletonPixels(individual_worm);

	# get endpoints of this skeleton
	endpoints = findEndpoints(individual_worm)

	if len(endpoints) == 2:
		# get coordinates in order to find the longest side
		coord = (endpoints[1][0] - endpoints[0][0],endpoints[1][1] - endpoints[0][1])
		# calculate hypotenuse
		hypotenuse = math.hypot(coord[1],coord[0])

		# calculate chance of it being a line
		straight_line_percentage = int(hypotenuse/pixels*100)
	elif len(endpoints) == 1:
		print("this worm is in a circle")
	elif len(endpoints) > 2:
		print("there is a cluster of worms")
		# print("- #worms in blob: ", math.floor(len(endpoints)/2))
		# # attempt to find intersection
		# print("- attempting to identify worm intersection")
		# intersectionpoints = findIntersections(individual_worm);
		# individual_worm = drawPoints(intersectionpoints, individual_worm)
		straight_line_percentage = -1
		# showDiagnosticImage("sk", individual_worm)
		# print(" - found intersection points:",len(intersectionpoints))
	
	# print("[" + str(worm_number) + "] - Worm Straight Line Confidence:", str(straight_line_percentage) + "%")
	return straight_line_percentage

def wormDOAPattern(worm,image,normalised):
	"""

	wormDOAPattern

	detects pattern difference between the worm and its blurred representation;

	@param worm: worm contour
	@param image: segmented image
	@param normalised: normalised image

	@return eucl_val: euclidean value (normalised to a 0-100 scale); difference between worm and its blurred version (the larger, the more different.)

	"""

	# create baseimage for mask
	mask_base = np.zeros(image.shape,np.uint8)
	cv2.drawContours(mask_base,[worm],0,255,-1)

	# apply mask to normalised
	check = cv2.bitwise_and(normalised, normalised, mask=mask_base)

	# single worm is spaced out.
	check_blurred = cv2.GaussianBlur(normalised,(15,15),0)
	check_blurred = cv2.bitwise_and(check_blurred, check_blurred, mask=mask_base)

	wow=np.concatenate((check, check_blurred), axis=1)
	# check = cv2.equalizeHist(check);

	check = cv2.equalizeHist(check);
	check_blurred = cv2.equalizeHist(check_blurred);

	check_vals = np.ndarray.flatten(check)/256
	blur_vals = np.ndarray.flatten(check_blurred)/256

	(img_height, img_width) = image.shape

	# calculate spatial distance; the larger the more similar.
	eucl_val = round(((spatial.distance.euclidean(check_vals, blur_vals)*10)),2)

	return eucl_val

#####################################################################

# Functions Related to Skeleton Manipulation

def findEndpoints(skeleton):
	"""

	findEndpoints

	finds endpoints of a skeleton

	@param skeleton: skeleton of worm
	@return skel_coords: array of tuples representing the coordinates of the endpoints of the skeleton.

	"""


	(rows,cols) = np.nonzero(skeleton)
	# Initialize empty list of co-ordinates
	skel_coords = []

	for (r,c) in zip(rows,cols):
		counter = countNeighbouringPixels(skeleton, r,c)
		if counter == 1:
			skel_coords.append((r,c))
	return skel_coords

def drawPoints(points,shape):

	"""
	
	drawPoints

	draws randomly coloured points in an image.

	@param points: list of 2-tuple points
	@param image: an image
	@return rgb_img: image with coloured points

	"""
	rgb_img = np.float32(shape)
	# print(rgb_img)
	rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_GRAY2RGB)
	colour = randomColourTuple()
	for i in points:
		rgb_img[i[0]][i[1]] = colour
		# cv2.circle(backtorgb, (i[1],i[0]),  3, colour, -1)
	# print(i)
	return rgb_img

def findIntersections(skeleton):
	"""
	
	findIntersections

	traverses through the skeleton, looking for pixels with more than 4 neighbours. If this is the case than it is considered to be an intersection in the skeleton, where worms are touching or coliding with each other.

	@param skeleton: skeleton image
	@return skel_coords: list of coordinates where a collision occurs.
	"""

	(rows,cols) = np.nonzero(skeleton)

	# Initialize empty list of co-ordinates
	skel_coords = []

	for (r,c) in zip(rows,cols):
		counter = countNeighbouringPixels(skeleton, r,c)
		if counter >= 4:
			skel_coords.append((r,c))
	return skel_coords

def countNeighbouringPixels(skeleton,x,y):
	"""

	countNeighbouringPixels

	gets the neighbours of a pixel in a skeleton.

	@param x: x coordinate
	@param y: y coordinate
	@param skeleton: skeleton image
	@return number of neighbours that have a 1 value. 

	"""
	# get neighbours
	neighbours = Neighbours(x,y,skeleton)
	return sum(neighbours)/255

def countSkeletonPixels(skeleton):	
	"""

	countSkeletonPixels

	counts number of pixels in a skeleton.

	@param skeleton: skeleton image
	@return pixels: # of pixels in skeleton.
	
	"""
	pixels = 0
	for i in range(0, len(skeleton)):
		for j in range(0, len(skeleton[0])):
			if skeleton[i][j] != 0:
				pixels += 1
	return pixels

def Neighbours(x,y,img):
	"""

	Neighbours

	returns neighbours of an image point in a clockwise order. top left, to left.

	@param x: x coordinate
	@param y: y coordinate
	@param skeleton: skeleton image
	@return list of values of a neighbouring pixel. (starts topleft, clockwise to left.)
	
	"""
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
			 img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]

#####################################################################

# Functions Related to Image Saving/Loading

def saveImage(filename, image, location):
	"""
	saveImage

	saves an image.

	@param filename: name of the file
	@param image: image content to be saved
	@param location: directory where image is to be saved

	"""
	checkDirectoryExists(location)
	save_file = location+filename+".png"
	cv2.imwrite(save_file, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 100])
	print("file is saved to",save_file)

def loadImage(filename, location):
	"""
	loadImage

	loads an image.

	@param filename: name of the file
	@param location: directory where image is to be loaded
	@return image, or false; depending on whether the image was found.

	"""
	save_file = location+filename+".png"
	image = cv2.imread(save_file);
	if not image is None:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# print(len(image), len(image[0]))
		return image
	else:
		return False

def saveWormToImage(column, number, worm, image, location, status):
	"""
	saveWormToImage

	saves an individual worm as an image file.

	@param column: name of the column
	@param number: worm number
	@param worm: worm contour
	@param image: thresholded image
	@param location: image location
	@param status: string to check whether worm is dead or not
	
	"""
	file_string = column + " - " + str(number);
	if status == "dead":
		file_string = file_string + " - Dead"
	# print(file_string)
	# create baseimage for mask
	mask_base = np.zeros(image.shape,np.uint8)
	cv2.drawContours(mask_base,[worm],0,255,-1)

	# apply mask to normalised
	worm_img = cv2.bitwise_and(image, image, mask=mask_base)

	# single worm is spaced out.
	saveImage(file_string, worm_img, location)

#####################################################################

# Functions used for Thresholding and Sanitising

def doUpperThresholding(image, threshold, value):
	"""
	doUpperThresholding

	thresholds an image

	@param image: raw image
	@param threshold: threshold value
	@param value: value to threshold by
	@return image: image post thresholding
	
	"""
	for i in range(0, len(image)):
		for j in range(0, len(image[i])):
			if image[i][j] > threshold:
				image[i][j] = value
	return image

def doLowerThresholding(image, threshold, value):
	"""
	doLowerThresholding

	thresholds an image

	@param image: raw image
	@param threshold: threshold value
	@param value: value to threshold by
	@return image: image post thresholding
	
	"""
	for i in range(0, len(image)):
		for j in range(0, len(image[i])):
			if image[i][j] < threshold:
				image[i][j] = value
	return image

def createThreshold(normalised):
	"""
	createThreshold

	performs various thresholding techniques on a normalised iamge

	@param normalised: normalised image
	@param worm_img: thresholded image
	
	"""

	# make a copy of the normalised
	worm_img = normalised.copy()
	# create a mask
	mask_s = worm_img.copy()

	# blur and threshold the image
	worm_img = cv2.GaussianBlur(worm_img,(9,9),0)
	worm_img = cv2.adaptiveThreshold(worm_img, 255,cv2.ADAPTIVE_THRESH_MEAN_C,\
			cv2.THRESH_BINARY,15,3)

	# create the mask
	mask_s = doUpperThresholding(mask_s, 20, 255)
	mask_s = doLowerThresholding(mask_s, 20, 0)
	element = np.ones((10,10),np.uint8)
	mask_s = cv2.erode(mask_s,element,iterations = 1)
	# mask = cv2.dilate(mask,element, iterations = 2)

	# invert the image so we can perform operations with the mask
	worm_img = cv2.bitwise_not(worm_img)
	# merge mask and image together
	worm_img= cv2.min(worm_img, mask_s)

	return worm_img

def normaliseImg(raw):
	"""
	normaliseImg

	normalise the 16 bit tif file, and fits it in a 8 bit image.

	@param raw: 16 bit image
	@param raw: 8 bit image
	
	"""

	# normalise the 16 bit tif file
	cv2.normalize(raw, raw, 0, 65535, cv2.NORM_MINMAX)
	# fit it back into a 8 bit image so we can work with it
	raw = (raw / 256).astype('uint8')
	return raw

def ThresholdW1(worm_img):
	"""
	ThresholdW1

	performs various thresholding techniques on a normalised W1 image

	@param worm_img: normalised image
	@param worm_img: thresholded image
	
	"""

	worm_img = (worm_img / 256).astype('uint8')
	worm_img = cv2.GaussianBlur(worm_img,(5,5),0)
	ret, worm_img = cv2.threshold(worm_img,0,255,cv2.THRESH_BINARY)
	worm_img = ParticleCleansing(worm_img)
	return worm_img

def ParticleCleansing(image):
	"""
	ParticleCleansing

	Searches for particles in an images and removes it using contours.

	@param image: thresholded image
	@param image: sanitised image
	
	"""

	# print("--- Initiating Cleansing ---")
	# find contours in image so we can look for particle
	_, contours, _= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	worm_count = 0
	for worm in contours:
		area = cv2.contourArea(worm)
		if area > 300:
			# it's a worm, don't mess with it
			worm_count += 1
		else:
			# its most likely a particle, colour it black.
			cv2.drawContours(image,[worm],0,0,-1)
	# print("Number of Worm Blobs Detected: " + str(worm_count))
	# print("Number of Particles Detected: " + str(len(contours)-worm_count))
	# print("--- Ending Cleansing ---")
	return image

#####################################################################

def coveragePercentage(raw, truth):
	"""
	coveragePercentage

	Searches for particles in an images and removes it using contours.

	@param raw: computed image
	@param truth: truth image
	@return percentage: percentage of computed image compared to truth.
	
	"""

	raw_pixels = 0
	tru_pixels = 0

	for i in range(0, len(raw)):
		for j in range(0, len(raw[0])):
			if raw[i][j] != 0:
				raw_pixels += 1
			if truth[i][j] != 0:
				tru_pixels += 1

	return int(raw_pixels / tru_pixels * 100)

# Functions used for Diagnostic Purposes

def showDiagnosticImage(title,image):
	"""
	showDiagnosticImage

	displays image

	@param title: title
	@param image:  image

	"""

	keep_processing = True;
	while (keep_processing):
		cv2.imshow(title, image);
		key = cv2.waitKey(40) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
		if (key == ord('x')):
			keep_processing = False;
	cv2.destroyAllWindows();

def writeDataResults(csv_file,location):
	"""
	writeDataResults

	writes data into a csv.

	@param csv_file:  dictionary list of results data
	@param location:  location for csv to be saved

	"""
	localtime = time.asctime( time.localtime(time.time()) )
	checkDirectoryExists(location);
	file_location = location + str(localtime) + ".csv"
	with open(file_location, "w") as text_file:
		for i in csv_file:
			print(i, file=text_file)
	print("file written to " + file_location)

def checkDirectoryExists(location):
	# checks whether a directory exists, if it doesn't; make it.
	if not os.path.exists(location):
		os.makedirs(location)

def randomColourTuple():
	# generates a random colour touple, self explanatory
	return (random.random() * 255, random.random() * 255, random.random() * 255)

#####################################################################

# Functions used for Display purposes

def writeOnWormImage(image,worm,value):
	# draws worm numbers on the screen so you can identify worms

	# calculate moment for worm
	M = cv2.moments(worm)

	# get center of image
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# initiate font
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image,str(value),(cX,cY), font, 0.6,(0,165,255),1,cv2.LINE_AA)
	# get center of contour so we can put position of numbers
	return image

def mapSkeletonOnImage(skeleton, image, skeleton_colour):
	"""
	mapSkeletonOnImage

	writes data into a csv.

	@param skeleton: skeleton of image
	@param image: thresholded image
	@return image: image with skeleton mapped on

	"""
	skeleton = cv2.cvtColor(skeleton,cv2.COLOR_GRAY2RGB) * skeleton_colour
	image = cv2.add(image, skeleton)
	return image
