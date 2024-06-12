# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np #numerical operations and handling arrays
# from sklearn.utils.linear_assignment_ import linear_assignment    # deprecated
from scipy.optimize import linear_sum_assignment #implementation of the hungarain algorithm for solvin ghte assignment problem, used for associating detections to trackers
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, iou3d #Function from a custom library (AB3DMOT_libs) for 3d bounding box converison and clacualtion intersection over Union (IoU)
from AB3DMOT_libs.kalman_filter import KalmanBoxTracker #A class from a custom library ('Ab3DMOT_libs.kalman_filter) implementing a kalman filter for tracking

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.01):   #Takes three arguments
	"""
	Assigns detections to tracked object (both represented as bounding boxes)

	detections:  N x 8 x 3
	trackers:    M x 8 x 3


	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	#If there are no trackers, the function returns an aempty array for matches, an array of indices for all detections (as all detections are unmatched), an empty array for unmatched trackers
	#Checks for existing tracked objects that were maintained by AB3DMOT and were tracked in the previous frame. If non it provides us with empty array 
	if (len(trackers)==0): 
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)    
	iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32) #Initializes the IoU matrix to store the IoU values between each detetion and tracker. The matrix has a shape of (number of detections, number of trackers)
	

    #Iterates over each detection and tracker
	#For each detection-tracker pair, computes the IoU and stors it in the iou_matrix
	# The iou3d function returns a tuple, and '[0]' extracts the IoU value
	for d, det in enumerate(detections):
		for t, trk in enumerate(trackers):
			iou_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
	# matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm, compatible to linear_assignment in sklearn.utils
	

    #Uses the linear_sum_assignment function to solve the assignment problem. It minimizes the total cost (negative Iou in this case) to find the best matches
	row_ind, col_ind = linear_sum_assignment(-iou_matrix)  
	matched_indices = np.stack((row_ind, col_ind), axis=1) #Combines the row indices ('row_ind') and colum indices ('col_ind') into a single array of matched detection-tracker pairs

    #Creates a list of indices for detections that are not in the matched pairs
	unmatched_detections = []
	for d, det in enumerate(detections):
		if (d not in matched_indices[:, 0]): unmatched_detections.append(d)
	
    #Creates a list of indices for trackers that are not in the matched pairs
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

	#Iterates through the matched pairs and checks if their IoU is below the threshold
	# Adds low Iou matches to the unmatched detectiosn and trackers
	#Adds valid matches (IoU above threshold to the matches list)
	matches = []
	for m in matched_indices:
		if (iou_matrix[m[0], m[1]] < iou_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else: matches.append(m.reshape(1, 2))
	
    #If there are no valid matches, returns an empty array.
	#Otherwise, concatenates the mathces into a single array
	if (len(matches) == 0): 
		matches = np.empty((0, 2),dtype=int)
	else: matches = np.concatenate(matches, axis=0)
	
    #Returns three arrays
	#mathces: Array of matched detection-tracker pairs
	#unmatched_detections: Array of indices for unmatched detections
	#unmatched_trackers: Array of indices for unmatched trackers

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):

    #Initialize the tracker with default parameters			  
	def __init__(self, max_age=2, min_hits=3):
		"""
		Sets key parameters for SORT                
		"""
		self.max_age = max_age #How long to keep track without updates
		self.min_hits = min_hits #Minimum number of hits before considering a track valid
		self.trackers = [] #List of current trackers
		self.frame_count = 0 #Number of frames processed
		
        #Index mapping to reorder detection data
		self.reorder = [3, 4, 5, 6, 2, 1, 0] # Order to convert [h, w, l, x, y, z, theta] to [x, y, z, theta, l, w, h]
		self.reorder_back = [6, 5, 4, 0, 1, 2, 3] # Order to convert back from [x, y, z, theta, l, w, h] to [h, w, l, x, y, z, theta]

	def update(self, dets_all):
		"""
		Params:
		  dets_all: dict
			dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
			info: a array of other info for each det
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		#Takes dictionary values from dets_all which is passed from the main.py script and 
		#Assigns it to dets (3d detections) and info (array of additional information)
		dets, info = dets_all['dets'], dets_all['info'] 

		#Reorders the 3d detections array so that it is the expected input for the kalman filter ([h, w, l, x, y, z, theta] to [x, y, z, theta, l, w, h])
		dets = dets[:, self.reorder]					
		self.frame_count += 1 #Increments frame count by 1 because its processing the first frame
		
        #trks is initially an empty array as self.trackers is any empty list
		#Each element in this list will be an instance of KalmanBoxTracker whcih represents an actively tracked object (detection)
		#When new detections are received, unmatched detectiosn are used to create a new KalmanBoxTracker isntance which
		# is then added to the self.trackers list
		#For each frame, existing trackers are updated with new detections. This involves prediciting the new state of a detection
		# of each tracker and then updating the state if a matching detection is found
		
        #For each frame, the predict method of the KalmanBoxTracker is called to estimate the new state of a previosuly
		# tracked object (detection). The prediction is made using the kalman filter, which uses the object's previous state
		# and a motion model to predict its new state. The predicted state will include the object's estimated position, velocity, and other attributes
		#This means that if there were 5 detections in the first frame, there would be 5 instances of the KalmanBoxTracker
		
        #Creates a 2d array of zeros with shape corresponding to the number of trackers and 7 state variables 
		#So if there were 5 detections in the first frame then there would be a matrix of 5 x 7 (5 3d detections)
		trks = np.zeros((len(self.trackers), 7)) #Initialize an array 'trks' to store the predicted states of all current trackers         
		to_del = [] #Initialize empty to list to keep track of indices of trackers that need to be deleted (invalid states)
		ret = [] #Initialize an empty list 'ret' to store the results (updated tracker states that will be returned at the end of the update method)
		for t, trk in enumerate(trks): #Start a lopp to iterate over each tracker and its corresponding index
			pos = self.trackers[t].predict().reshape((-1, 1)) #Predict the new state of the tracker using the kalman filter - Prediction may return a shape that needs reshaping
			trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]] #Updates the trks array with the predicted state for the current tracker (5 detections in one frame, trks array will have 5 new predicted detections for frame 2)       
			if (np.any(np.isnan(pos))): #Check if the predicted state contains any Not a Number values (undefined value during the mathematical operation)
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) #Remove rows in the trks array that contain invali values (NaN), compress the array by removing masked rows (those with NaNs) 
		for t in reversed(to_del): #Remove trackers that have invalid predicted states
			self.trackers.pop(t)
			
        #We are given the 3d detection in a compact format but need to find the actual coordinates of the 8 corners for calculations

		dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets] #Convert each 3d bounding box in 'dets' to an 8-corner representation. (the function does this specifically). Result is a list of 8-corner representations, 'dets_8corner'
		
        #Convert the list of 8-corner detections into a numpy array.
		#Checks if the list of 8-corner detections is not empty.
		#Stacks the list of 8-corner arrays into a single Numpy array with shape (N, 8, 3).
		#N is number of detections, '8' represents the 8 cornersof the 3d box and 3 represent the 3d coordinates of each corner
		if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		else: dets_8corner = []
		
        #Converts each 3d bounding box in trks (predicted states) to an 8-corner representation
		trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks] 
		#Convert the list of 8-corner trackers into a numpy array
		if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
		
        #The associate_detection_to_trackers function in this line is what actually matches new detections to exisitng trackers based on their predicte states
		#The hungarian algorithm is typically used to minimize the total cost and find the best matches 
		#Inputs are the 8-corner representation of the current frame's detections
		#and the 8 corner representation of the current tracker's predicted state
		#matched is an array of matched pairs of detection indices and tracker indices
		#unmatched_dets is an array of indices for detections that could not be matched to any tracker
		#unmatched_trks is an array of indices for trackers that could not be matched to any detections
		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)

		# For each matched detection-tracker pair, the 'update' method of the KalmanBoxTracker is called to refine the tracker's state with the new detection
		# This invloves using the new detection to correct the predicted state, imporving its accuracy
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]
				trk.update(dets[d, :][0], info[d, :][0])

		# create and initialize new trackers for unmatched detections
		# new detections are likely new detections that appeared in the seen therefore new trackers need to be created for them using the kalman filter
		for i in unmatched_dets:        # a scalar of index
			trk = KalmanBoxTracker(dets[i, :], info[i, :]) 
			self.trackers.append(trk)
		i = len(self.trackers)
		
        #Iterates over all updated trackers and retrieves their current state and reorder the stae vector to a specified format
		for trk in reversed(self.trackers):
			d = trk.get_state()      # bbox location
			d = d[self.reorder_back]			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			
            #formats array to be in the right form with the id, additional info and the 3d detection that has been tracked
			#id comes from the kalman filter as each instance is assigned a new id
			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
				ret.append(np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1)) # +1 as MOT benchmark requires positive
			i -= 1

			# remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(i)
		if (len(ret) > 0): return np.concatenate(ret)			# resulting output: h,w,l,x,y,z,theta, ID, other info, confidence
		return np.empty((0, 15))    