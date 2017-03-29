#####################################################################

# Image Processing Summative
# cmkv68 - T. Nguyen
# Durham University

"""

-------- README ---------

To run this file, first include the worm images in the same directory
as this file (including ground truths). For example:

Directory:
	- BBBC010_v1_images/
	- BBBC010_v1_foreground/
	- BBBC010_v1_foreground_eachworm/
	- base_functions.py
	- run.py
	- zhang.py
	- cmkv68.pdf

If opencv needs to be run before use, then run in the console:

	opencv3-1.init

Then/otherwise, run the following command:

	python3 run.py

The terminal will populate with information accordingly. Worms are not
shown during the processing, but images will populate in the directory
while this script is run. Information about the processing will also 
display on the terminal. In the event that the script terminates for
whatever reason, Results data will be writen before python temrinates
this script.

Results images will be saved in the $results_img_dir directory. By
default, that will be in the directory:

	- data/results_img/

Results information are saved as a `.csv` file wherever specified
in the $data_csv_dir variable on line 58. By default, that will be
in the directory:

	- data/results_data/


-------- README ---------

"""

#####################################################################


# Input Locations

# This is where the ground truth and raw images
# will be loaded. By default, they are set to be loaded
# from the folders that are created after immediately unzipping
# the images into the base directory of where this (run.py) is
# found.

gt_groups_dir 			= "BBBC010_v1_foreground/";
gt_individuals_dir 		= "BBBC010_v1_foreground_eachworm/";
raw_file_dir 			= "BBBC010_v1_images/";


# Output Locations

# This is where the images will be saved.
# The directory can be changed accordingly.
# Folders will be made if they are not created.

results_img_dir			= "data/results_img/"
skeleton_img_dir		= "data/skeleton/";
thresh_results_dir		= "data/thresholdresults/";
normalised_img_dir		= "data/normalised/";
individual_worms_dir	= "data/individual_worms/";
data_csv_dir			= "data/results_data/";


# Don't attempt to modify the code below this line!

#####################################################################
#####################################################################
#####################################################################
#####################################################################

# import libaries
import zhang as zs
import base_functions as bf
import cv2
import glob, os
import re
import atexit

# Set up Variables
columns = {};
raw_images = [];
results_csv = [];

def writeOnExit():
	"""
	Exit Handler
	Writes results into a .csv file.
	"""
	print('Worm Checking Ending; Attempt to write results to .CSV')
	if data_csv:
		results_csv.insert(0,", ".join([ str(v) for v in data_csv.keys() ]))
		bf.writeDataResults(results_csv, data_csv_dir)
	else:
		print("Script quit too early in order to retrieve results.")

atexit.register(writeOnExit)

#####################################################################

class ImageSet:
	def __init__(self,input):
		"""
		ImageSet class
		will contain overall information about a column, including raw images,
		number of worms (from reference) and associated file data.
		"""
		self.column = input[len(gt_groups_dir):len(gt_groups_dir)+3];
		self.binary = input;
		self.raw_w1 = "";
		self.raw_w2 = "";
		# self.worm_count = 0;
		self.gt_worms = [];
		# self.column = 
	def wormCount(self):
		return len(self.gt_worms)

class ImageFile:
	"""
	image file class;
	will contain information about the image loaded from the /raw folder
	"""
	def __init__(self, input):
		temp = imagefile[len(raw_file_dir) + 33:];
		temp = temp.split("_");
		self.column = temp[0];
		self.wavelength = temp[1];
		self.raw_fileid = temp[2][:-4];

		# determine control type from the column
		ct = int(re.search(r'\d+', self.column).group())
		if ct > 12:
			self.controltype = "untreated negative"
		else:
			self.controltype = "treated positive"

	def getFilename(self):
		out = False
		if self.column and self.wavelength and self.raw_fileid:
			out = "1649_1109_0003_Amp5-1_B_20070424_" + self.column + "_" + self.wavelength + "_" + self.raw_fileid + ".tif";
		return out;

	def getNameData(self):
		return (self.column, self.wavelength)

#####################################################################

# go to /ground_truths/groups/
# set up columns

for image in glob.glob(gt_groups_dir + "*.png"):
	col = image[len(gt_groups_dir):len(gt_groups_dir)+3]
	columns[col] = ImageSet(image)

#####################################################################


# count the number of worms for each col in /ground_truths/individuals/

for image in glob.glob(gt_individuals_dir + "*.png"):
	strbase = len(gt_individuals_dir)
	# print(image)
	col = image[strbase:strbase+3]
	# print("col",col)
	count = image[strbase+4:strbase+6]
	# print("count", count)
	columns[col].gt_worms.append(image)

#####################################################################

# go to dir containing the raw images, loop through all images
for imagefile in glob.glob(raw_file_dir + '*.tif'):
	raw_imagefile = ImageFile(imagefile)
	raw_images.append(raw_imagefile)

# at this stage, image details are loaded onto the images list.

#####################################################################

# loop through images, allocate to columns.

for img in raw_images:
	if img.wavelength == 'w1':
		columns[img.column].raw_w1 = img
	elif img.wavelength == 'w2':
		columns[img.column].raw_w2 = img

#####################################################################

def performImageProcessing(col_in_work):
	"""
	Performs Image Processing Functions on a column.

	@param col_in_work: column of image
	@return: data_csv; array of data retrieved from calculating.
	"""

	# initiate data_csv file for each column
	data_csv = {}
	data_csv['Column'] = col_in_work

	w1 = None;
	w2 = None;

	try:
		read_this1 = columns[col_in_work].raw_w1.getFilename();
		w1 = cv2.imread(raw_file_dir + read_this1, -1);
	except:
		print("can't load w1")

	try:
		read_this2 = columns[col_in_work].raw_w2.getFilename();
		w2 = cv2.imread(raw_file_dir + read_this2, -1);
	except:
		print("can't load w2")


	if not w2 is None:
		print(col_in_work, "loaded")
		if not w1 is None:
			data_csv['Loaded'] = "w1 & w2"
		else:
			data_csv['Loaded'] = "w2"

		# make a copy of the normalised image

		normalised = bf.loadImage(col_in_work, normalised_img_dir)
		if normalised is False:
			normalised = bf.normaliseImg(w2);

			# save the normalised image so we don't have to create it again.

			bf.saveImage(col_in_work, normalised, normalised_img_dir)
		data_csv['Normalised'] = True


		# load sanitised raw data if it exists.

		check = bf.loadImage(col_in_work, thresh_results_dir)
		if check is False:

			# sanitise and filter the image so we can have a working file

			w2 = bf.createThreshold(normalised);

			# detect particles/artefacts so we can remove them
			w2 = bf.ParticleCleansing(w2)

			if not w1 is None:

				# threshold w1, will be used as a support image

				w1 = bf.ThresholdW1(w1);

			img = cv2.bitwise_or(w2, w1)

			# save the image so we don't have to create it again.

			bf.saveImage(col_in_work, img, thresh_results_dir)		
		else:
			img = check
		data_csv['Sanitised'] = True

		# load ground_truth

		group_truth = cv2.imread(columns[col_in_work].binary, 0);

		# compare results of data created with the ground truth

		data_csv['Coverage Percentage'] = bf.coveragePercentage(img, group_truth)

		# load skeleton if it exists.

		skeleton = bf.loadImage(col_in_work, skeleton_img_dir)
		if skeleton is False:

			# create skeleton by providing a binary threshold as input

			skeleton = zs.zhangSuen(img)

			# save skeleton to file so we don't have to bother processing it again

			bf.saveImage(col_in_work, skeleton, skeleton_img_dir)
		data_csv['Skeleton'] = True


		# at this stage the image should be ready to work with.


		# get contours of the present worms

		_, wormblobs, _= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# perform worm check algorithm

		bf.wormDOACheck(data_csv, 
						img, 
						wormblobs, 
						normalised, 
						skeleton, 
						individual_worms_dir,
						results_img_dir)

		# added ground truth worm count

		data_csv['Ground Truth Worm Count'] = columns[col_in_work].wormCount()


		# calculate percentage

		no_worms = data_csv['Worms Found']
		wormcount_percentage = int(no_worms/columns[col_in_work].wormCount()*100)
		data_csv['Worm Count Percentage'] = wormcount_percentage

		no_worms_cl = data_csv['Worms Found (Clusters)']
		wormcount_percentage = int(no_worms_cl/columns[col_in_work].wormCount()*100)
		data_csv['Worm Count Percentage (Clusters)'] = wormcount_percentage
	else:
		print("can't load", col_in_work)
		data_csv['Loaded'] = False

	# add results to results_csv
	return data_csv

print("Number of Columns Loaded:", len(columns))

for col_in_work in columns:
# col_in_work = 'A01'
	data_csv = performImageProcessing(col_in_work)
	results_csv.append(", ".join([ str(v) for v in data_csv.values() ]))

print("Worm images have finished processing.")