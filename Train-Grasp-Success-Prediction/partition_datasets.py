import numpy as np
import os
import sys
import time

class Gripper:
	PARALLEL      = 0
	UNDERACTUATED = 1

class BOPObject:
	LMO1   = 0
	LMO5   = 1
	LMO6   = 2
	LMO8   = 3
	LMO9   = 4
	LMO10  = 5
	LMO11  = 6
	LMO12  = 7
	YCBV2  = 8
	YCBV3  = 9
	YCBV4  = 10
	YCBV5  = 11
	YCBV8  = 12
	YCBV9  = 13
	YCBV12 = 14

class PoseDifferenceMode:
	TRANS_ERR_SCALAR_ROT_ERR_SCALAR           = 0					#  Store translation error as a scalar and rotation error as a scalar.
	TRANS_ERR_3VEC_ROT_ERR_3VEC               = 1					#  Store translation error as a SIGNED 3-vec(x, y, z) and rotation error as a SIGNED 3-vec(axis-angle minus axis).
	TRANS_ERR_SCALAR_ROT_ERR_3VEC             = 2					#  Store translation error as a scalar and rotation error as a SIGNED 3-vec(axis-angle minus axis).
	TRANS_ERR_3VEC_ROT_ERR_SCALAR             = 3					#  Store translation error as a SIGNED 3-vec(x, y, z) and rotation error as a scalar.
	AVERAGE_DISTANCE_DISTINGUISHABLE          = 4					#  Store the average distance (single scalar) between corresponding points in each pose.
	WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE = 5					#  Store the (weighted) average distance (single scalar) between corresponding points in each pose.

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['principal-estimator'] is None or params['object'] is None or params['gripper'] is None or params['helpme']:
		usage()
		return
																	#  Require existence of these sources.
	root_file_name = 'dataset-' + params['principal-estimator'] + '-' + params['mode-str']

	if not (os.path.exists(root_file_name + '.txt') and os.path.exists(root_file_name + '.npz')):
		print('ERROR: Unable to find "' + root_file_name + '.txt" and/or "' + root_file_name + '.npz".')
		return

	case = {}														#  key: (DatasetObject-ID, Success-Case) ==> val: [ indices into src_txt ]
																	#    for Success-Case in {Parallel-Success, Parallel-Fail}
																	#                  or in {Underactuated-Success, Underactuated-Fail}
	fh_src_txt = open(root_file_name + '.txt', 'r')
	src_txt = []													#  File content--without header.
	estimators_armed = False
	datasets_armed = False
	src_txt_index = 0
	for line in fh_src_txt.readlines():
		if line[0] == '#':
			if '#  SUPPORT ESTIMATOR(S):' in line:					#  Retrieve the estimators used to build this dataset.
				estimators_armed = True
			elif '#  DATASET(S):' in line:							#  Retrieve the datasets used to build this dataset.
				datasets_armed = True
			elif estimators_armed:
				estimators = line[1:].replace('{', '').replace('}', '').replace(',', '').strip().split()
				principal_est_combos = [params['principal-estimator'] + '-' + x for x in estimators]
				estimators_armed = False
			elif datasets_armed:
				datasets = line[1:].replace('{', '').replace('}', '').replace(',', '').strip().split()
				datasets_armed = False
		else:
			arr = line.strip().split('\t')
			dataset = arr[0]										#  Identify dataset.
			object_id = arr[3]										#  Identify dataset-local object ID.
			object_str = dataset + object_id

			if object_str == params['object-str']:

				if params['gripper'] == Gripper.PARALLEL:
					if (object_str, 'Parallel-Success') not in case:
						case[ (object_str, 'Parallel-Success') ] = []
					if (object_str, 'Parallel-Fail') not in case:
						case[ (object_str, 'Parallel-Fail') ] = []
				elif params['gripper'] == Gripper.UNDERACTUATED:
					if (object_str, 'Underactuated-Success') not in case:
						case[ (object_str, 'Underactuated-Success') ] = []
					if (object_str, 'Underactuated-Fail') not in case:
						case[ (object_str, 'Underactuated-Fail') ] = []

				scene = int(arr[1])
				frame = int(arr[2])
				par_success = (arr[-2] == '1')
				und_success = (arr[-1] == '1')

				if params['gripper'] == Gripper.PARALLEL:
					if par_success:
						case[ (object_str, 'Parallel-Success') ].append( src_txt_index )
					else:
						case[ (object_str, 'Parallel-Fail') ].append( src_txt_index )
				elif params['gripper'] == Gripper.UNDERACTUATED:
					if und_success:
						case[ (object_str, 'Underactuated-Success') ].append( src_txt_index )
					else:
						case[ (object_str, 'Underactuated-Fail') ].append( src_txt_index )

				src_txt.append( line )								#  Build persistent reference to file body.
				src_txt_index += 1
	fh_src_txt.close()

	if params['verbose']:											#  Display statistics.
		file_name = root_file_name + '.txt'
		print(file_name)
		print('='*len(file_name))
		max_str_len = len(params['object-str'])

		if params['gripper'] == Gripper.PARALLEL:
			print('    '+' '*max_str_len+'  Par.Succ.    Par.Fail     TOTAL')
			print('    '+params['object-str']+' '*(max_str_len - len(params['object-str']))+'  '+\
			                  str(len(case[ (params['object-str'], 'Parallel-Success') ])).rjust(4, ' ')+\
			      '         '+str(len(case[ (params['object-str'], 'Parallel-Fail') ])).rjust(4, ' ')+\
			      '         '+str(len(case[ (params['object-str'], 'Parallel-Success') ])+len(case[ (params['object-str'], 'Parallel-Fail') ])).rjust(4, ' '))
		elif params['gripper'] == Gripper.UNDERACTUATED:
			print('    '+' '*max_str_len+'  Und.Succ.    Und.Fail     TOTAL')
			print('    '+params['object-str']+' '*(max_str_len - len(params['object-str']))+'  '+\
			      '         '+str(len(case[ (params['object-str'], 'Underactuated-Success') ])).rjust(4, ' ')+\
			      '         '+str(len(case[ (params['object-str'], 'Underactuated-Fail') ])).rjust(4, ' ')+\
			      '         '+str(len(case[ (params['object-str'], 'Underactuated-Success') ])+len(case[ (params['object-str'], 'Underactuated-Fail') ])).rjust(4, ' '))

	timestring = time.strftime('%l:%M%p %Z on %b %d, %Y')			#  Single time stamp for all files.

	fh_dst_train_txt = open('-'.join(['train', params['principal-estimator'], params['object-str'], params['gripper-str'], params['mode-str']])+'.txt', 'w')
	fh_dst_train_txt.write('#  Training set for an ensemble grasp-success predictor.\n')
	fh_dst_train_txt.write('#  Created ' + timestring + '\n')
	fh_dst_train_txt.write('#    python3 '+' '.join(sys.argv)+'\n')
	fh_dst_train_txt.write('#  PRINCIPAL ESTIMATOR:' + '\n')
	fh_dst_train_txt.write('#    ' + params['principal-estimator'] + '\n')
	fh_dst_train_txt.write('#  SUPPORT ESTIMATOR(S):' + '\n')
	fh_dst_train_txt.write('#    {' + ', '.join(estimators) + '}\n')
	fh_dst_train_txt.write('#  OBJECT:' + '\n')
	fh_dst_train_txt.write('#    ' + params['object-str'] + '\n')
	fh_dst_train_txt.write('#  DATASET(S):' + '\n')
	fh_dst_train_txt.write('#    {' + ', '.join(datasets) + '}\n')
	fh_dst_train_txt.write('#  GRIPPER:' + '\n')
	fh_dst_train_txt.write('#    ' + params['gripper-str'] + '\n')
	fh_dst_train_txt.write('#  MODE:\n')
	fh_dst_train_txt.write('#    '+params['mode-str']+'\n')
	fh_dst_train_txt.write('#\n')
	fh_dst_train_txt.write('#  Indices refer to a single NumPy file, "' + root_file_name + '.npz", that contains all support estimators\' differences against the principal.\n')
	fh_dst_train_txt.write('#\n')
	fh_dst_train_txt.write('#  LINE FORMAT' + '\n')
	fh_dst_train_txt.write('#  Dataset <t> Scene <t> Frame <t> Object-ID <t> Visibility <t> ')
	fh_dst_train_txt.write(' <t> '.join(['Idx.diff.' + x for x in principal_est_combos]) + ' <t> Success('+params['principal-estimator']+')\n')
	total_train = []												#  Collection of lines.

	fh_dst_test_txt = open('-'.join(['test', params['principal-estimator'], params['object-str'], params['gripper-str'], params['mode-str']])+'.txt', 'w')
	fh_dst_test_txt.write('#  Test set for an ensemble grasp-success predictor.\n')
	fh_dst_test_txt.write('#  Created ' + timestring + '\n')
	fh_dst_test_txt.write('#    python3 '+' '.join(sys.argv)+'\n')
	fh_dst_test_txt.write('#  PRINCIPAL ESTIMATOR:' + '\n')
	fh_dst_test_txt.write('#    ' + params['principal-estimator'] + '\n')
	fh_dst_test_txt.write('#  SUPPORT ESTIMATOR(S):' + '\n')
	fh_dst_test_txt.write('#    {' + ', '.join(estimators) + '}\n')
	fh_dst_test_txt.write('#  OBJECT:' + '\n')
	fh_dst_test_txt.write('#    ' + params['object-str'] + '\n')
	fh_dst_test_txt.write('#  DATASET(S):' + '\n')
	fh_dst_test_txt.write('#    {' + ', '.join(datasets) + '}\n')
	fh_dst_test_txt.write('#  GRIPPER:' + '\n')
	fh_dst_test_txt.write('#    ' + params['gripper-str'] + '\n')
	fh_dst_test_txt.write('#  MODE:\n')
	fh_dst_test_txt.write('#    '+params['mode-str']+'\n')
	fh_dst_test_txt.write('#\n')
	fh_dst_test_txt.write('#  Indices refer to a single NumPy file, "' + root_file_name + '.npz", that contains all support estimators\' differences against the principal.\n')
	fh_dst_test_txt.write('#\n')
	fh_dst_test_txt.write('#  LINE FORMAT' + '\n')
	fh_dst_test_txt.write('#  Dataset <t> Scene <t> Frame <t> Object-ID <t> Visibility <t> ')
	fh_dst_test_txt.write(' <t> '.join(['Idx.diff.' + x for x in principal_est_combos]) + ' <t> Success('+params['principal-estimator']+')\n')
	total_test = []													#  Collection of lines.

	if params['shuffle']:											#  Shuffle?
		if params['gripper'] == Gripper.PARALLEL:
			for success_case in ['Parallel-Success', 'Parallel-Fail']:
				key = (params['object-str'], success_case)
				np.random.shuffle( case[key] )						#  Shuffles in place.
		elif params['gripper'] == Gripper.UNDERACTUATED:
			for success_case in ['Underactuated-Success', 'Underactuated-Fail']:
				key = (params['object-str'], success_case)
				np.random.shuffle( case[key] )						#  Shuffles in place.

	train = []														#  [ (list of lines into src_txt, success-case-string) ]
	test = []														#  [ (list of lines into src_txt, success-case-string) ]

	if params['gripper'] == Gripper.PARALLEL:
		for success_case in ['Parallel-Success', 'Parallel-Fail']:
			key = (params['object-str'], success_case)
			alloc_train = int(round(len(case[key]) * params['training-portion']))
			alloc_test  = len(case[key]) - alloc_train

			i = 0
			while i < alloc_train:
				train.append( (case[key][i], success_case) )
				i += 1
			while i < len(case[key]):
				test.append( (case[key][i], success_case) )
				i += 1
	elif params['gripper'] == Gripper.UNDERACTUATED:
		for success_case in ['Underactuated-Success', 'Underactuated-Fail']:
			key = (params['object-str'], success_case)
			alloc_train = int(round(len(case[key]) * params['training-portion']))
			alloc_test  = len(case[key]) - alloc_train

			i = 0
			while i < alloc_train:
				train.append( (case[key][i], success_case) )
				i += 1
			while i < len(case[key]):
				test.append( (case[key][i], success_case) )
				i += 1

	for index_case in train:
		index = index_case[0]
		sample = index_case[1]
		arr = src_txt[index].strip().split('\t')[:-2]
		if sample == 'Parallel-Success':
			arr += [1]
		elif sample == 'Parallel-Fail':
			arr += [0]
		elif sample == 'Underactuated-Success':
			arr += [1]
		elif sample == 'Underactuated-Fail':
			arr += [0]
		total_train.append('\t'.join([str(x) for x in arr])+'\n')

	for index_case in test:
		index = index_case[0]
		sample = index_case[1]
		arr = src_txt[index].strip().split('\t')[:-2]
		if sample == 'Parallel-Success':
			arr += [1]
		elif sample == 'Parallel-Fail':
			arr += [0]
		elif sample == 'Underactuated-Success':
			arr += [1]
		elif sample == 'Underactuated-Fail':
			arr += [0]
		total_test.append('\t'.join([str(x) for x in arr])+'\n')

	#if params['shuffle']:											#  Shuffle?
	#	np.random.shuffle( total_train )							#  Shuffles in place.
	#	np.random.shuffle( total_test )								#  Shuffles in place.

	for line in total_train:										#  Write-out training.
		fh_dst_train_txt.write(line)

	for line in total_test:											#  Write-out test.
		fh_dst_test_txt.write(line)

	fh_dst_train_txt.close()
	fh_dst_test_txt.close()

	#################################################################  Report.
	stats_train = {}												#  key: Success-Case ==> val: ctr
	stats_test = {}													#    for Success-Case in {Success, Fail}
	stats_train['Success'] = 0
	stats_train['Fail'] = 0
	stats_test['Success'] = 0
	stats_test['Fail'] = 0

	fh = open('-'.join(['train', params['principal-estimator'], params['object-str'], params['gripper-str'], params['mode-str']])+'.txt', 'r')
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			if arr[-1] == '1':
				stats_train['Success'] += 1
			elif arr[-1] == '0':
				stats_train['Fail'] += 1
	fh.close()

	fh = open('-'.join(['test', params['principal-estimator'], params['object-str'], params['gripper-str'], params['mode-str']])+'.txt', 'r')
	for line in fh.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			if arr[-1] == '1':
				stats_test['Success'] += 1
			elif arr[-1] == '0':
				stats_test['Fail'] += 1
	fh.close()

	if params['verbose']:											#  Display statistics.
		print('\nTRAINING-SET')
		if params['gripper'] == Gripper.PARALLEL:
			print('    '+' '*max_str_len+'  Par.Succ.    Par.Fail     TOTAL')
		elif params['gripper'] == Gripper.UNDERACTUATED:
			print('    '+' '*max_str_len+'  Und.Succ.    Und.Fail     TOTAL')
		print('    '+params['object-str']+' '*(max_str_len - len(params['object-str']))+'  '+\
		                  str(stats_train[ 'Success' ]).rjust(4, ' ')+\
		      '         '+str(stats_train[ 'Fail' ]).rjust(4, ' ')+\
		      '         '+str(stats_train[ 'Success' ]+stats_train[ 'Fail' ]).rjust(4, ' '))

		print('\nVALIDATION-SET')
		if params['gripper'] == Gripper.PARALLEL:
			print('    '+' '*max_str_len+'  Par.Succ.    Par.Fail     TOTAL')
		elif params['gripper'] == Gripper.UNDERACTUATED:
			print('    '+' '*max_str_len+'  Und.Succ.    Und.Fail     TOTAL')
		print('    '+params['object-str']+' '*(max_str_len - len(params['object-str']))+'  '+\
		                  str(stats_test[ 'Success' ]).rjust(4, ' ')+\
		      '         '+str(stats_test[ 'Fail' ]).rjust(4, ' ')+\
		      '         '+str(stats_test[ 'Success' ]+stats_test[ 'Fail' ]).rjust(4, ' '))

	return

def get_command_line_params():
	params = {}

	params['principal-estimator'] = None

	params['object'] = None
	params['object-str'] = None
	params['bop-objects'] = ['LMO1', 'LMO5', 'LMO6', 'LMO8', 'LMO9', 'LMO10', 'LMO11', 'LMO12', 'YCBV2', 'YCBV3', 'YCBV4', 'YCBV5', 'YCBV8', 'YCBV9', 'YCBV12']

	params['gripper'] = None
	params['gripper-str'] = None

	params['mode'] = PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC
	params['mode-str'] = 't3r3'
	params['mode-args'] = ['t1r1', 't3r3', 't1r3', 't3r1']

	params['training-portion'] = 0.8								#  Default.
	params['shuffle'] = False										#  Default.

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-principal', '-object', '-gripper', '-mode', '-train', '-test', '-shuffle', '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-shuffle':
				params['shuffle'] = True
			elif sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-principal':
					params['principal-estimator'] = argval
				elif argtarget == '-object':
					if argval.upper().replace('-', '') == 'LMO1':
						params['object'] = BOPObject.LMO1
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO5':
						params['object'] = BOPObject.LMO5
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO6':
						params['object'] = BOPObject.LMO6
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO8':
						params['object'] = BOPObject.LMO8
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO9':
						params['object'] = BOPObject.LMO9
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO10':
						params['object'] = BOPObject.LMO10
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO11':
						params['object'] = BOPObject.LMO11
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'LMO12':
						params['object'] = BOPObject.LMO12
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV2':
						params['object'] = BOPObject.YCBV2
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV3':
						params['object'] = BOPObject.YCBV3
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV4':
						params['object'] = BOPObject.YCBV4
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV5':
						params['object'] = BOPObject.YCBV5
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV8':
						params['object'] = BOPObject.YCBV8
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV9':
						params['object'] = BOPObject.YCBV9
						params['object-str'] = argval.upper().replace('-', '')
					elif argval.upper() == 'YCBV12':
						params['object'] = BOPObject.YCBV12
						params['object-str'] = argval.upper().replace('-', '')
				elif argtarget == '-gripper':
					if argval.lower()[:3] == 'par':
						params['gripper'] = Gripper.PARALLEL
						params['gripper-str'] = 'Parallel'
					elif argval.lower()[:3] == 'und':
						params['gripper'] = Gripper.UNDERACTUATED
						params['gripper-str'] = 'Underactuated'
				elif argtarget == '-mode':
					if argval.lower() == 't1r1':
						params['mode'] = PoseDifferenceMode.TRANS_ERR_SCALAR_ROT_ERR_SCALAR
						params['mode-str'] = argval.lower()
					elif argval.lower() == 't3r3':
						params['mode'] = PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC
						params['mode-str'] = argval.lower()
					elif argval.lower() == 't1r3':
						params['mode'] = PoseDifferenceMode.TRANS_ERR_SCALAR_ROT_ERR_3VEC
						params['mode-str'] = argval.lower()
					elif argval.lower() == 't3r1':
						params['mode'] = PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_SCALAR
						params['mode-str'] = argval.lower()
				elif argtarget == '-train':
					if float( argval ) < 1.0 and float( argval ) > 0.0:
						params['training-portion'] = float( argval )
				elif argtarget == '-test':
					if float( argval ) < 1.0 and float( argval ) > 0.0:
						params['training-portion'] = 1.0 - float( argval )

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('This is the third step.')
	print('Allocate samples in "dataset-<principal>.txt/dataset-<principal>.npz" for training and for test.')
	print('This script allocates for both grippers.')
	print('Script ensures comparable distributions of the object categories.')
	print('')
	print('Usage:  python3 partition_datasets.py <parameters, preceded by flags>')
	print(' e.g.:  python3 partition_datasets.py -principal EPOS -object LMO1 -gripper par -train 0.8 -shuffle -mode t3r3 -v')
	print(' e.g.:  python3 partition_datasets.py -principal GDRNPP -object LMO1 -gripper par -train 0.8 -shuffle -mode t3r3 -v')
	print(' e.g.:  python3 partition_datasets.py -principal ZebraPose -object LMO1 -gripper par -train 0.8 -shuffle -mode t3r3 -v')
	print('')
	print('Flags:  -principal  REQUIRED: Identify which data collection to partition.')
	print('        -object     REQUIRED: In {LMO1,LMO5,LMO6,LMO8,LMO9,LMO10,LMO11,LMO12,YCBV2,YCBV3,YCBV4,YCBV5,YCBV8,YCBV9,YCBV12}.')
	print('        -gripper    REQUIRED: In {par, und}.')
	print('        -mode       In {t1r1, t3r3, t1r3, t3r1}. Default is t3r3.')
	print('                    Determines how a pose discrepancy is stored/given to network.')
	print('        -train      Following real in (0.0, 1.0) is the portion of data to allocate to the training set. Default is 0.8.')
	print('        -test       Following real in (0.0, 1.0) is the portion of data to allocate to the test set. Default is 0.2.')
	print('        -shuffle    Shuffle the data set before allocating.')
	print('        -v          Enable verbosity.')
	print('        -?          Display this message.')
	return

if __name__ == '__main__':
	main()
