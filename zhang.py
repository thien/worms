"""
A modified version of Linbo Jin's Implementation of the Zhang-Suen Algorithm.
The original can be found here:
https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
"""

import numpy as np
import time
import cv2
import base_functions as bf

def neighbours(x,y,image):
	"""Return 8-neighbours of image point P1(x,y), in a clockwise order"""
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],	 # P2,P3,P4,P5
				img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]	# P6,P7,P8,P9

def transitions(neighbour, white):
	count = 0
	"""No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"""
	n = neighbour + neighbour[0:1]	  # P2, P3, ... , P8, P9, P2
	return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
	"""the Zhang-Suen Thinning Algorithm"""
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	white = 1
	layer = 0
	time_start = time.time()
	skeleton = image.copy()  # deepcopy to protect the original image

	skeleton = skeleton / 255

	changing1 = changing2 = 1		#  the points to be removed (set as 0)
	while changing1 or changing2:   #  iterates until no further changes occur in the image
		layer += 1
		print("ZS: Layer " , layer)
		changing1 = []
		rows, columns = skeleton.shape			   # x for rows, y for columns
		for x in range(1, rows - 1):					 # No. of  rows
			for y in range(1, columns - 1):			# No. of columns
				n = neighbours(x, y, skeleton)
				P2,P3,P4,P5,P6,P7,P8,P9 = n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7]

				if skeleton[x][y] == white: # Condition 0: Point P1 in the object regions 
					# print(x,y, "is white")
					if 2 <= np.count_nonzero(n) <= 6:	# Condition 1: 2<= N(P1) <= 6
						if transitions(n, white) == 1:	# Condition 2: S(P1)=1  
							if P2 * P4 * P6 == 0:	# Condition 3   
								if P4 * P6 * P8 == 0:		 # Condition 4
									changing1.append((x,y))
		for x, y in changing1: 
			# print("setting ", x,y, "to be 0")
			skeleton[x][y] = 0
		# Step 2
		changing2 = []
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, skeleton)
				if (skeleton[x][y] == white   and		# Condition 0
					2 <= np.count_nonzero(n) <= 6  and	   # Condition 1
					transitions(n, white) == 1 and	  # Condition 2
					P2 * P4 * P8 == 0 and	   # Condition 3
					P2 * P6 * P8 == 0):			# Condition 4
					changing2.append((x,y))	
		for x, y in changing2: 
			skeleton[x][y] = 0
	skeleton = skeleton * 255

	print("ZS: Time Taken: ", time.time() - time_start)
	return skeleton