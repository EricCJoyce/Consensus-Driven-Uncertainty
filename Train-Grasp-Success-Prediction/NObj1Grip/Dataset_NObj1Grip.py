import numpy as np
import os
import re
import sys
import torch
from torch.utils.data import Dataset

'''
Class for a dataset for several objects and a single gripper.
'''
class Dataset_NObj1Grip(Dataset):
	def __init__(self, files):
		self.len_ctr = 0											#  Count up how many samples this data set contains.

		X = []														#  For each pair of estimators, difference of pose estimates
		y = []														#  Grasp success indicator in {0, 1}
		self.src = []												#  Let's afford ourselves the ability to quickly look up which cases led to errors.
		self.objects_encountered = []								#  Store unique object strings, as taken from file names.

		max_prog_bar_len = os.get_terminal_size().columns			#  Total width of the terminal.

		for file in files:											#  FIRST pass: encounter ALL OBJECTS IN SET DETERMINED BY "files".
			head, tail = os.path.split(file)						#  "head" is the path leading up to "tail", the file.
																	#  [train, test]-<principal estimator>-<object>-<gripper>-<mode>.txt
			arr = tail.split('.')[0].split('-')						#  Will reveal whether or not this data set is object-specific.
			principal_estimator = arr[1]							#  Always [1], in {EPOS, GDRNPP, ZebraPose}.
			obj = arr[2]											#  Always [2], in {LMO1, LMO5, LMO6, LMO8, LMO9, LMO10, LMO11, LMO12, YCBV2, YCBV3, YCBV4, YCBV5, YCBV8, YCBV9, YCBV12}.
			gripper = arr[3]										#  Always [3], in {Parallel, Underactuated}.
			mode = arr[4]											#  Always [4], in {t1r1, t3r3, t1r3, t3r1, t4r4, add, wadd}.
			if obj not in self.objects_encountered:
				self.objects_encountered.append( obj )				#  Mark as recognized. In the order given.

		self.one_hot_length = len(self.objects_encountered)			#  Lock down the length of the one-hot sub-vector.

		for file in files:											#  SECOND pass: encounter ALL OBJECTS IN SET DETERMINED BY "files".
			head, tail = os.path.split(file)						#  "head" is the path leading up to "tail", the file.
																	#  [train, test]-<principal estimator>-<object>-<gripper>-<mode>.txt
			arr = tail.split('.')[0].split('-')						#  Will reveal whether or not this data set is object-specific.
			principal_estimator = arr[1]							#  Always [1], in {EPOS, GDRNPP, ZebraPose}.
			obj = arr[2]											#  Always [2], in {LMO1, LMO5, LMO6, LMO8, LMO9, LMO10, LMO11, LMO12, YCBV2, YCBV3, YCBV4, YCBV5, YCBV8, YCBV9, YCBV12}.
			gripper = arr[3]										#  Always [3], in {Parallel, Underactuated}.
			mode = arr[4]											#  Always [4], in {t1r1, t3r3, t1r3, t3r1, t4r4, add, wadd}.
																	#  Load the NumPy part: dataset-<principal estimator>-<mode>.npz
			npz = np.load(os.path.join(head, 'dataset-'+principal_estimator+'-'+mode+'.npz'))
			print('Loading "'+file+'"')
			fh = open(file, 'r')
			lines = [x for x in fh.readlines() if x[0] != '#']
			ctr = 0
			prev_ctr = 0
			num_lines = len(lines)
			fh.close()

			for line in lines:
				arr = line.strip().split('\t')
				dataset = arr[0]									#  [0] = Dataset
				scene = arr[1]										#  [1] = Scene
				frame = arr[2]										#  [2] = Frame
				local_id = arr[3]									#  [3] = Object-ID
				visibility = arr[4]									#  [4] = Visibility
				indices = tuple( [int(x) for x in arr[5:-1]] )		#  [5] = Idx.diff.<principal>-<supporter-1>
																	#  [6] = Idx.diff.<principal>-<supporter-2>
				success = float(arr[-1])							#  [7] = Success(<principal>)
				principal_support_differences = []
				for index in indices:
					if mode == 't1r1':								#  From a 1-by-2.
						principal_support_differences += list(npz['arr_0'][index].reshape((2, )))
					elif mode == 't3r3':							#  From a 3-by-2.
						principal_support_differences += list(npz['arr_0'][index][:, 0].reshape((3, ))) + list(npz['arr_0'][index][:, 1].reshape((3, )))
					elif mode == 't1r3':							#  From a 1-by-4.
						principal_support_differences += list(npz['arr_0'][index].reshape((4, )))
					elif mode == 't3r1':							#  From a 1-by-4.
						principal_support_differences += list(npz['arr_0'][index].reshape((4, )))
					elif mode == 't4r4':							#  From a 4-by-2.
						principal_support_differences += list(npz['arr_0'][index][:, 0].reshape((4, ))) + list(npz['arr_0'][index][:, 1].reshape((4, )))
					elif mode == 'add':								#  From a 1-by-1.
						principal_support_differences += list(npz['arr_0'][index].reshape((1, )))
					elif mode == 'wadd':							#  From a 1-by-1.
						principal_support_differences += list(npz['arr_0'][index].reshape((1, )))
																	#  Save reference to input size.
				self.input_length = len(principal_support_differences) + self.one_hot_length

				one_hot_vector = [0.0 for i in range(0, self.one_hot_length)]
				one_hot_vector[ self.objects_encountered.index( obj ) ] = 1.0
																	#  To be reshaped (batch-size x input_length).
				X.append( principal_support_differences + one_hot_vector )
				y.append( success )									#  To be reshaped (batch-size x 1).
																	#  Save reference, so we can look up the source for this sample.
				self.src.append( (dataset, scene, frame, obj, visibility, gripper) )
				self.len_ctr += 1

				prog_output_str = ' ' + str(int(np.ceil(float(ctr) / float(num_lines) * 100.0))) + '%]'
																	#  Leave enough space for the brackets, space, and percentage
				prev_ctr = int(np.ceil(float(ctr) / float(num_lines) * float(max_prog_bar_len - len(prog_output_str)))) - 2
				sys.stdout.write('\r[' + '-'*prev_ctr + prog_output_str)
				sys.stdout.flush()
				ctr += 1

			prog_output_str = ' 100%]'
			prev_ctr = max_prog_bar_len - len(prog_output_str) - 2
			sys.stdout.write('\r[' + '-'*prev_ctr + prog_output_str)
			sys.stdout.flush()
			print('')												#  Clear the line.

		self.X = torch.tensor( X, dtype=torch.float32 )				#  Convert to Torch Tensors.
		self.y = torch.tensor( y, dtype=torch.float32 )

	def __len__(self):
		return self.len_ctr

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()
		return self.X[index], self.y[index]
