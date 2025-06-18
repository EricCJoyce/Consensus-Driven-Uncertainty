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
	if params['helpme']:
		usage()
		return

	max_str_len = max(len(x) for x in params['bop-objects'])

	stats_total = {}												#  key: principal ==> val: { key: gripper ==> val: { key: object ==> val: { Success ==> 0,
	stats_train = {}												#                                                                           Fail ==> 0}
	stats_test = {}													#                          }                       }

	for estimator in ['EPOS', 'GDRNPP', 'ZebraPose']:
		stats_total[ estimator ] = {}
		stats_train[ estimator ] = {}
		stats_test[ estimator ]  = {}
		for gripper in ['Parallel', 'Underactuated']:
			stats_total[ estimator ][ gripper ] = {}
			stats_train[ estimator ][ gripper ] = {}
			stats_test[ estimator ][ gripper ]  = {}
			for obj in params['bop-objects']:
				stats_total[ estimator ][ gripper ][ obj ] = {}
				stats_total[ estimator ][ gripper ][ obj ]['Success'] = 0
				stats_total[ estimator ][ gripper ][ obj ]['Fail']    = 0

				stats_train[ estimator ][ gripper ][ obj ] = {}
				stats_train[ estimator ][ gripper ][ obj ]['Success'] = 0
				stats_train[ estimator ][ gripper ][ obj ]['Fail']    = 0

				stats_test[ estimator ][ gripper ][ obj ]  = {}
				stats_test[ estimator ][ gripper ][ obj ]['Success'] = 0
				stats_test[ estimator ][ gripper ][ obj ]['Fail']    = 0

	for estimator in ['EPOS', 'GDRNPP', 'ZebraPose']:
		for gripper in ['Parallel', 'Underactuated']:
			for obj in params['bop-objects']:
				fh = open('-'.join(['train', estimator, obj, gripper, params['mode-str']])+'.txt', 'r')
				for line in fh.readlines():
					if line[0] != '#':
						arr = line.strip().split('\t')
						if arr[-1] == '1':
							stats_total[ estimator ][ gripper ][ obj ]['Success'] += 1
							stats_train[ estimator ][ gripper ][ obj ]['Success'] += 1
						elif arr[-1] == '0':
							stats_total[ estimator ][ gripper ][ obj ]['Fail'] += 1
							stats_train[ estimator ][ gripper ][ obj ]['Fail'] += 1
				fh.close()

				fh = open('-'.join(['test', estimator, obj, gripper, params['mode-str']])+'.txt', 'r')
				for line in fh.readlines():
					if line[0] != '#':
						arr = line.strip().split('\t')
						if arr[-1] == '1':
							stats_total[ estimator ][ gripper ][ obj ]['Success'] += 1
							stats_test[ estimator ][ gripper ][ obj ]['Success']  += 1
						elif arr[-1] == '0':
							stats_total[ estimator ][ gripper ][ obj ]['Fail'] += 1
							stats_test[ estimator ][ gripper ][ obj ]['Fail']  += 1
				fh.close()

	fh = open('data-split-profile.txt', 'w')
	for estimator in ['EPOS', 'GDRNPP', 'ZebraPose']:
		file_name = 'dataset-'+estimator+'-'+params['mode-str']+'.txt'
		fh.write(file_name+'\n')
		fh.write('='*len(file_name)+'\n')
		fh.write('    '+' '*max_str_len+'  Par.Succ.    Par.Fail     TOTAL\n')
		for obj in params['bop-objects']:
			fh.write('    '+obj+' '*(max_str_len - len(obj))+'  '+\
			                    str(stats_total[estimator]['Parallel'][obj]['Success']).rjust(4, ' ')+\
			        '         '+str(stats_total[estimator]['Parallel'][obj]['Fail']).rjust(4, ' ')+\
			        '         '+str(stats_total[estimator]['Parallel'][obj]['Success']+stats_total[estimator]['Parallel'][obj]['Fail']).rjust(4, ' ')+'\n')
		fh.write('    '+' '*max_str_len+'  Und.Succ.    Und.Fail     TOTAL\n')
		for obj in params['bop-objects']:
			fh.write('    '+obj+' '*(max_str_len - len(obj))+'  '+\
			                     str(stats_total[estimator]['Underactuated'][obj]['Success']).rjust(4, ' ')+\
			         '         '+str(stats_total[estimator]['Underactuated'][obj]['Fail']).rjust(4, ' ')+\
			         '         '+str(stats_total[estimator]['Underactuated'][obj]['Success']+stats_total[estimator]['Underactuated'][obj]['Fail']).rjust(4, ' ')+'\n')

		fh.write('\nTRAINING-SET\n')
		fh.write('    '+' '*max_str_len+'  Par.Succ.    Par.Fail     TOTAL\n')
		for obj in params['bop-objects']:
			fh.write('    '+obj+' '*(max_str_len - len(obj))+'  '+\
			                     str(stats_train[estimator]['Parallel'][obj]['Success']).rjust(4, ' ')+\
			         '         '+str(stats_train[estimator]['Parallel'][obj]['Fail']).rjust(4, ' ')+\
			         '         '+str(stats_train[estimator]['Parallel'][obj]['Success']+stats_train[estimator]['Parallel'][obj]['Fail']).rjust(4, ' ')+'\n')
		fh.write('    '+' '*max_str_len+'  Und.Succ.    Und.Fail     TOTAL\n')
		for obj in params['bop-objects']:
			fh.write('    '+obj+' '*(max_str_len - len(obj))+'  '+\
			                     str(stats_train[estimator]['Underactuated'][obj]['Success']).rjust(4, ' ')+\
			         '         '+str(stats_train[estimator]['Underactuated'][obj]['Fail']).rjust(4, ' ')+\
			         '         '+str(stats_train[estimator]['Underactuated'][obj]['Success']+stats_train[estimator]['Underactuated'][obj]['Fail']).rjust(4, ' ')+'\n')

		fh.write('\nVALIDATION-SET\n')
		fh.write('    '+' '*max_str_len+'  Par.Succ.    Par.Fail     TOTAL\n')
		for obj in params['bop-objects']:
			fh.write('    '+obj+' '*(max_str_len - len(obj))+'  '+\
			                     str(stats_test[estimator]['Parallel'][obj]['Success']).rjust(4, ' ')+\
			         '         '+str(stats_test[estimator]['Parallel'][obj]['Fail']).rjust(4, ' ')+\
			         '         '+str(stats_test[estimator]['Parallel'][obj]['Success']+stats_test[estimator]['Parallel'][obj]['Fail']).rjust(4, ' ')+'\n')
		fh.write('    '+' '*max_str_len+'  Und.Succ.    Und.Fail     TOTAL\n')
		for obj in params['bop-objects']:
			fh.write('    '+obj+' '*(max_str_len - len(obj))+'  '+\
			                     str(stats_test[estimator]['Underactuated'][obj]['Success']).rjust(4, ' ')+\
			         '         '+str(stats_test[estimator]['Underactuated'][obj]['Fail']).rjust(4, ' ')+\
			         '         '+str(stats_test[estimator]['Underactuated'][obj]['Success']+stats_test[estimator]['Underactuated'][obj]['Fail']).rjust(4, ' ')+'\n')
	fh.close()

	return

def get_command_line_params():
	params = {}

	params['bop-objects'] = ['LMO1', 'LMO5', 'LMO6', 'LMO8', 'LMO9', 'LMO10', 'LMO11', 'LMO12', 'YCBV2', 'YCBV3', 'YCBV4', 'YCBV5', 'YCBV8', 'YCBV9', 'YCBV12']

	params['mode'] = PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC
	params['mode-str'] = 't3r3'
	params['mode-args'] = ['t1r1', 't3r3', 't1r3', 't3r1']

	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-mode', '-v', '-?', '-help', '--help']
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
				if argtarget == '-mode':
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
	return params

def usage():
	print('python3 profile_data_split.py -mode t3r3')
	return

if __name__ == '__main__':
	main()