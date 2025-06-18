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

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['principal'] is None or params['object'] is None or params['gripper'] is None or params['helpme']:
		usage()
		return

	train = []														#  List of tuples: [ (Disagreement, Grasp-Success), .... ]
																	#  Ascending by "Disagreement."
	fh_train = open(os.path.join('..', 'DataSplit', 'add', '-'.join(['train', params['principal'], params['object-str'], params['gripper-str'], 'add']) + '.txt'), 'r')
	npz = np.load(os.path.join('..', 'DataSplit', 'add', '-'.join(['dataset', params['principal'], 'add']) + '.npz'))
	for line in fh_train.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')							#  Dataset <t> Scene <t> Frame <t> Object-ID <t> Visibility <t> Idx.diff.1 <t> Idx.diff.2 <t> Success(Principal)
			dataset      = arr[0]
			scene        = arr[1]
			frame        = arr[2]
			object_id    = arr[3]
			visibility   = arr[4]
			index_diff_1 = int(arr[5])
			index_diff_2 = int(arr[6])
			success      = int(arr[7])
			diff1        = npz['arr_0'][ index_diff_1 ]
			diff2        = npz['arr_0'][ index_diff_2 ]
			train.append( (np.mean([diff1, diff2]), success) )
	fh_train.close()

	#################################################################  Determine the optimal threshold for this (principal, object, gripper).
	train = sorted(train, key=lambda x: x[0])						#  Sort ascending by disagreement value.

	best_accuracy = 0.0
	best_threshold = 0.0
	for i in range(0, len(train)):									#  Allow iteration the possibly of encompassing ALL samples.
		if i == len(train) - 1:
			threshold = float('inf')
		else:
			threshold = train[i + 1][0]
		M = np.zeros((2, 2), dtype=np.int16)
		for j in range(0, len(train)):
			if train[j][0] < threshold:
				pred = 1
			else:
				pred = 0
			gt = train[j][1]
			M[gt, pred] += 1
		acc = float(M.trace()) / float(np.sum(M))
		if acc > best_accuracy:
			best_accuracy = acc
			best_threshold = threshold

	#################################################################  Evaluate the threshold for this (principal, object, gripper) on the test set.
	test = []
	test_guide = []

	fh_test = open(os.path.join('..', 'DataSplit', 'add', '-'.join(['test', params['principal'], params['object-str'], params['gripper-str'], 'add']) + '.txt'), 'r')
	for line in fh_test.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')							#  Dataset <t> Scene <t> Frame <t> Object-ID <t> Visibility <t> Idx.diff.1 <t> Idx.diff.2 <t> Success(Principal)
			dataset      = arr[0]
			scene        = arr[1]
			frame        = arr[2]
			object_id    = arr[3]
			visibility   = arr[4]
			index_diff_1 = int(arr[5])
			index_diff_2 = int(arr[6])
			success      = int(arr[7])
			diff1        = npz['arr_0'][ index_diff_1 ]
			diff2        = npz['arr_0'][ index_diff_2 ]

			test.append( (np.mean([diff1, diff2]), success) )
			test_guide.append( (dataset, scene, frame, visibility, params['gripper-str']) )
	fh_test.close()

	test = sorted(test, key=lambda x: x[0])							#  Sort ascending by disagreement value so we can see where the threshold falls.
	M = np.zeros((2, 2), dtype=np.int16)
	threshold_crossed = False

	timestamp_str = time.strftime('%l:%M%p %Z on %b %d, %Y')

	fh = open('Baseline-'+params['principal']+'-'+params['object-str']+'-'+params['gripper-str']+'-record.txt', 'w')
	fh.write('#  BASELINE method predictions on test set.\n')
	fh.write('#  PRINCIPAL:   '+params['principal']+'\n')
	fh.write('#  OBJECT(S):  {'+params['object-str']+'}\n')
	fh.write('#  GRIPPER(S): {'+params['gripper-str']+'}\n')
	fh.write('#  MODE:        add (meters)\n')
	fh.write('#  Created ' + timestamp_str + '.\n')
	fh.write('#    python3 ' + ' '.join(sys.argv) + '\n')
	fh.write('#  LINE FORMAT:' + '\n')
	fh.write('#    Prediction <t> Ground-truth label. <t> Dataset <t> Scene <t> Frame <t> Visibility <t> Gripper\n')

	prediction_lines = []

	ctr = 0
	for sample in test:
		x = sample[0]
		gt = sample[1]
		if x < best_threshold:
			pred = 1
		else:
			pred = 0

		fh.write('\t'.join([str(pred), str(gt)] + list(test_guide[ctr])) + '\n')

		M[gt, pred] += 1
		if params['verbose']:
			if x > best_threshold and not threshold_crossed:
				print('<--------th.-------->')
				threshold_crossed = True
			print(str(x) + '\t' + str(pred) + '\t' + str(gt))

		ctr += 1

	fh.close()

	if params['verbose'] and not threshold_crossed:
		print('<--------th.-------->')

	acc = float(M.trace()) / float(np.sum(M))

	fh = open('Baseline-'+params['principal']+'-'+params['object-str']+'-'+params['gripper-str']+'-confusionmatrix.txt', 'w')
	fh.write('#  Confusion matrix for BASELINE method.\n')
	fh.write('#  PRINCIPAL:   '+params['principal']+'\n')
	fh.write('#  OBJECT(S):  {'+params['object-str']+'}\n')
	fh.write('#  GRIPPER(S): {'+params['gripper-str']+'}\n')
	fh.write('#  MODE:        add (meters)\n')
	fh.write('#  ACCURACY:    '+str(acc)+'\n')
	fh.write('#  Created ' + timestamp_str + '.\n')
	fh.write('#    python3 ' + ' '.join(sys.argv) + '\n')
	fh.write('#  Rows: ground truth; Columns: predictions.\n')
	fh.write('#  0\t1\n')
	fh.write('0\t'+str(M[0, 0])+'\t'+str(M[0, 1])+'\n')
	fh.write('1\t'+str(M[1, 0])+'\t'+str(M[1, 1])+'\n')
	fh.close()

	if params['verbose']:
		print('Test set accuracy of threshold('+str(best_threshold)+') = ' + str(acc))
		print(M)

	return

def get_command_line_params():
	params = {}

	params['principal'] = None										#  Principal estimator.

	params['object'] = None
	params['object-str'] = None

	params['gripper'] = None
	params['gripper-str'] = None

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-principal', '-object', '-gripper', '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-v':
				params['verbose'] = True
			elif sys.argv[i] == '-?' or sys.argv[i] == '-help' or sys.argv[i] == '--help':
				params['helpme'] = True
			else:
				argtarget = sys.argv[i]
		else:
			argval = sys.argv[i]

			if argtarget is not None:
				if argtarget == '-principal':
					params['principal'] = argval
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

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('Following Shi et al., "Fast Uncertainty Quantification for Deep Object Pose Estimation", 2021,')
	print('this script builds a dataset for a baseline method, assuming a PRINCIPAL POSE ESTIMATOR.')
	print('')
	print('Usage:  python3 baseline_1Obj1Grip.py <parameters, preceded by flags>')
	print(' e.g.:  python3 baseline_1Obj1Grip.py -principal EPOS -object LMO1 -gripper par -v')
	print(' e.g.:  python3 baseline_1Obj1Grip.py -principal GDRNPP -object LMO1 -gripper par -v')
	print(' e.g.:  python3 baseline_1Obj1Grip.py -principal ZebraPose -object LMO1 -gripper par -v')
	print('')
	print('Flags:  -principal  REQUIRED: Following 6-DoF pose estimator is the Principal Estimator.')
	print('        -object     REQUIRED: In {LMO1,LMO5,LMO6,LMO8,LMO9,LMO10,LMO11,LMO12,YCBV2,YCBV3,YCBV4,YCBV5,YCBV8,YCBV9,YCBV12}.')
	print('        -gripper    REQUIRED: In {par, und}.')
	print('        -v          Enable verbosity')
	print('        -?          Display this message')

if __name__ == '__main__':
	main()
