# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

if __name__ == '__main__':
	if len(sys.argv) != 2: #Checks if the number of command-line arguments passed to the script is exactly
		print('Usage: python main.py result_sha(e.g., pointrcnn_Car_test)') #If not 2 print this statement and exit the system
		sys.exit(1) #Exit the code

	result_sha = sys.argv[1] #Assigns first command line argument to the variable result_sha (pointrcnn_Car_test - the data set)
	save_root = './results' #Root directory for saving results - variable save_root is assigned the string ./results which is the path where the results will be saved
	
    # Detections are made with integers and then changed to human readable format afterwards
	det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'} #This line defines a dictionary that maps detection IDs to object types

    #Constructs a path by combining data/KITTI and the value of result_sha
	#Ex: data/KITTI/pointrcnn_Car_test
	#load_list_from_folder is a function (defined somewhere else) that takes a directory path and returns a list of
	#sequence files (seq_file_list) and the number of sequences
	#returns a list of the file names along with how many text files there are
	seq_file_list, num_seq = load_list_from_folder(os.path.join('data/KITTI', result_sha)) #Loads a list of sequence files and the number of sequences from a specified folder
	
	total_time, total_frames = 0.0, 0 #Initializes time taken to proces and total number of frames processed
	save_dir = os.path.join(save_root, result_sha); mkdir_if_missing(save_dir) #Contructs a directory path for saving results and ensures that the directory exists - creates a direcotry specified by save_dir if it doesn't already exist
	eval_dir = os.path.join(save_dir, 'data'); mkdir_if_missing(eval_dir) #Constructs path for evaluation data (/results/pointrcnn_car_test/data)
	seq_count = 0 #Initializes variable to keep track the number of sequences processed
	
    #Sets up data for processing
	
	for seq_file in seq_file_list: #Begins iterating over each file in 'seq_file_list'
		_, seq_name, _ = fileparts(seq_file) #Extracts the base name of the sequence file (seq_name = 0001 or seq_name = 0002)
		eval_file = os.path.join(eval_dir, seq_name + '.txt'); eval_file = open(eval_file, 'w') #Constructs path for the evalutation file and opens it for writing, blank files where the human readable results go (/results/pointrcnn_car_test/data/0001.txt) for evalutation after tracker has processed
		save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name); mkdir_if_missing(save_trk_dir) #Creates the path for the tracking results directory and ensures it exists (save_trk_dir = './results/pointrcnn_Car_test/trk_withid/seq_01')

		mot_tracker = AB3DMOT() #Initializes the 3d multi-object tracker from the class Ab3DMOT and assged to mot_tracker
		#loads the content of seq_file as a NumPy array, assuming that the values are separated by a comma
		#seq_dets is assged the loaded data, which is expected to be an array with shape N x 15  where N is number of detections
		seq_dets = np.loadtxt(seq_file, delimiter=',') # load detections from current sequnce file, N x 15
		
		# if no detection in a sequence
		if len(seq_dets.shape) == 1: seq_dets = np.expand_dims(seq_dets, axis=0) #Ensures 2d array even if only one detection
		#If no valid detections, closes evaluation file and moves to the next sequence 	
		if seq_dets.shape[1] == 0: #returns the shape of the array if there only 1 detection it will be (15,) and adds an extra dimension to make it a 2d array with shape (1, 15)
			eval_file.close()
			continue

		# loop over frame
		
        #Determines the range of frames in the sequence
		#seq_dets[:, 0] extracts the frame numbers from the detection data
		#min() and max() find the minimum and maximum frame numbers, respectivley
		min_frame, max_frame = int(seq_dets[:, 0].min()), int(seq_dets[:, 0].max())
		for frame in range(min_frame, max_frame + 1): #Iterate over each frame in the sequence from min_frame to max_frame (the loop goes through every frame number in the specified range)
			# logging
			#prints a string with current sequence name, sequence count, total number of sequences, current frame number, and maximum frame number
			print_str = 'processing %s: %d/%d, %d/%d   \r' % (seq_name, seq_count, num_seq, frame, max_frame) #print current processing status to the console
			sys.stdout.write(print_str) #print string
			sys.stdout.flush() #ensures its displayed immediately
			
            #Create and open a file for saving tracking results fro the current frame
			#Creates a path for the trackign files, using the current fram number formatted as a sixdigit number with leading zeros
			#Opens the file for writing
			save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame); save_trk_file = open(save_trk_file, 'w') #results/pointrcnn_car_test/0000 (sequence number)/000000.txt

			# extract relevant infromation from the detections for the current frame - we are working with the array of all detections now but frame by frame
			'''
			0,2,726.4030,173.5915,917.4820,315.0742,13.8527,1.5599,1.5848,3.4791,2.5702,1.5720,9.7190,-1.5595,-1.8180
            0,2,679.6621,174.1462,795.2264,260.7374,12.8676,1.5694,1.6609,4.1846,2.4745,1.6008,15.2377,-1.5807,-1.7417
            0,2,668.8994,172.5304,740.3965,234.3335,9.0682,1.6581,1.6439,3.9524,2.6547,1.6498,21.3426,-1.5576,-1.6813
            0,2,192.2586,142.5810,449.8517,338.8805,7.2070,1.9773,1.7422,4.1917,-3.9889,1.6727,9.3978,-2.3139,-1.9125
            0,2,657.2991,175.4645,712.7974,223.0470,6.8313,1.5826,1.6715,4.0609,2.6098,1.6854,26.2665,-1.5514,-1.6504
			'''
			
            #selects all rows where the first column (frame number) equals the current frame
			ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1)) #Selects orientation values for detections in the current frame and reshapes them into a column vector
			'''
			array([
                [-1.8180],
                [-1.7417],
                [-1.6813],
                [-1.9125],
                [-1.6504]
                ])
			'''
			other_array = seq_dets[seq_dets[:, 0] == frame, 1:7] #Selects other detection infromationsich as 2d bounding box coordinates
			'''
			array([
                [2, 726.4030, 173.5915, 917.4820, 315.0742, 13.8527],
                [2, 679.6621, 174.1462, 795.2264, 260.7374, 12.8676],
                [2, 668.8994, 172.5304, 740.3965, 234.3335, 9.0682],
                [2, 192.2586, 142.5810, 449.8517, 338.8805, 7.2070],
                [2, 657.2991, 175.4645, 712.7974, 223.0470, 6.8313]
                ])
			'''

			additional_info = np.concatenate((ori_array, other_array), axis=1) #Concatenates the orientation array and othe rinfromation along the second axis (columns)
			'''
			array([
                [-1.8180, 2, 726.4030, 173.5915, 917.4820, 315.0742, 13.8527],
                [-1.7417, 2, 679.6621, 174.1462, 795.2264, 260.7374, 12.8676],
                [-1.6813, 2, 668.8994, 172.5304, 740.3965, 234.3335, 9.0682],
                [-1.9125, 2, 192.2586, 142.5810, 449.8517, 338.8805, 7.2070],
                [-1.6504, 2, 657.2991, 175.4645, 712.7974, 223.0470, 6.8313]
                ])
			'''
            #Selects the 3d bounding box parameters for all detections in the current frame
			dets = seq_dets[seq_dets[:,0] == frame, 7:14]     
			dets_all = {'dets': dets, 'info': additional_info} #dictionary containg the 3d detections and additional information

			# update tracker with the current frame's detections and measure the time taken
			start_time = time.time() #record current time
			trackers = mot_tracker.update(dets_all) #updates the tracker with the detections for processing
			cycle_time = time.time() - start_time #calculates the elapsed time for this frame
			total_time += cycle_time #accumalates total processing time

			# Extract and organize tracking results for each tracked object
            # d is a tracked object, with various attributes stored in different parts of the array
            # from the array produced by the tracker it will extract specific attributes such as 3d bounding boc, track ID,
            # orientation, object type, 2d bounding box and confidence score			
			for d in trackers:
				bbox3d_tmp = d[0:7]       # h, w, l, x, y, z, theta in camera coordinate
				id_tmp = d[7]
				ori_tmp = d[8]
				type_tmp = det_id2str[d[9]]
				bbox2d_tmp_trk = d[10:14]
				conf_tmp = d[14]

				# save in detection format with track ID, can be used for dection evaluation and tracking visualization
				# constructs a string with the tracking results in the required format
				# write the string to the save_trk_file
				str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
					bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
					bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
				save_trk_file.write(str_to_srite)

				# save in tracking format, for 3D MOT evaluation
				str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
					type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
					bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], 
					conf_tmp)
				eval_file.write(str_to_srite)

			total_frames += 1 #Increment the total number of processed frames
			save_trk_file.close() #Close the current frame's tracking result file
		seq_count += 1 #Increment total number of processes sequences
		eval_file.close() #closes the current sequences evalutation file    
	print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))