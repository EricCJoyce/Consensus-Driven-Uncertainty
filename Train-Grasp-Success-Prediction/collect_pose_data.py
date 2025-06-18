from itertools import product
import numpy as np
import os
import sys
import time

def main():
	params = get_command_line_params()								#  Collect parameters.
	if len(params['estimators']) == 0 or len(params['datasets']) == 0 or params['helpme']:
		usage()
		return

	est_data_product = [x for x in product(params['estimators'], params['datasets'])]

	data = {}														#  key: estimator-name ==>
	for estimator in params['estimators']:							#    val: {
		data[estimator] = {}										#           key: (dataset, scene, frame, object) ==>
																	#             val: {
																	#                    key: 'visibility' ==> val: visibility,
																	#                    key: 'estimate'   ==> val: R^(4x4),
																	#                    key: 'success'    ==> val: (parallel success, underactuated success)
																	#                  }
																	#         }
	for est_data in est_data_product:
		estimator_name = est_data[0]								#  Shorthand.
		dataset_name = est_data[1]									#  Does this pair exist?
		if os.path.exists(os.path.join('Results', 'Pose-Estimates', estimator_name, dataset_name)):
			if params['verbose']:
				print('>>> Found data for (' + ' '.join([estimator_name, dataset_name]) + ')')
																	#  Build filepath for estimator and dataset.
			if est_data in params['estimator-weight-set']:
				source_path = os.path.join('Results', 'Pose-Estimates', estimator_name, dataset_name, params['estimator-weight-set'][ est_data ])
				parallel_path = os.path.join('Results', 'Parallel-Gripper', estimator_name, dataset_name, params['estimator-weight-set'][ est_data ])
				underactuated_path = os.path.join('Results', 'Underactuated-Gripper', estimator_name, dataset_name, params['estimator-weight-set'][ est_data ])
			else:
				source_path = os.path.join('Results', 'Pose-Estimates', estimator_name, dataset_name)
				parallel_path = os.path.join('Results', 'Parallel-Gripper', estimator_name, dataset_name)
				underactuated_path = os.path.join('Results', 'Underactuated-Gripper', estimator_name, dataset_name)

			parallel_files = sorted([x for x in os.listdir(parallel_path) if x.endswith('.txt')], key=lambda x: int(x.split('-')[1]))
			underactuated_files = sorted([x for x in os.listdir(underactuated_path) if x.endswith('.txt')], key=lambda x: int(x.split('-')[1]))

			parallel_object_ids = [x.split('-')[1] for x in parallel_files]
			underactuated_object_ids = [x.split('-')[1] for x in underactuated_files]
			assert parallel_object_ids == underactuated_object_ids, 'ERROR: Inconsistent trials!'

			object_ids = parallel_object_ids

			npz_files = sorted([x for x in os.listdir(source_path) if x.endswith('.npz') and x.split('-')[1] in object_ids], key=lambda x: int(x.split('-')[1]))
			index_files = sorted([x for x in os.listdir(source_path) if x.endswith('.txt') and x.split('-')[1] in object_ids], key=lambda x: int(x.split('-')[1]))

			for i in range(0, len(npz_files)):
				fh_txt = open(os.path.join(source_path, index_files[i]), 'r')
				fh_npz = np.load(os.path.join(source_path, npz_files[i]))
				object_id = npz_files[i].split('-')[1]

				num_trials = 0
				for line in fh_txt.readlines():
					if line[0] != '#':
						arr = line.strip().split('\t')
						if arr[2] != '*':							#  No estimate failures.
																	#  (dataset, scene, frame, object)
							if (dataset_name, int(arr[0]), int(arr[1]), int(object_id)) not in data[estimator_name]:
								data[estimator_name][ (dataset_name, int(arr[0]), int(arr[1]), int(object_id)) ] = {}
								data[estimator_name][ (dataset_name, int(arr[0]), int(arr[1]), int(object_id)) ]['visibility'] = arr[4]

							trial_index = int(arr[2]) + 1			#  Trial results are ONE-INDEXED!!!
																	#  Was the parallel gripper successful?
							parallel_fh = open(os.path.join(parallel_path, parallel_files[i]), 'r')
							parallel_lines = parallel_fh.readlines()
							j = 0
							while j < len(parallel_lines):
								parallel_arr = parallel_lines[j].strip().split('\t')
								if int(parallel_arr[0]) == trial_index:
									parallel_success = parallel_arr[1] == '1'
									break
								j += 1
							parallel_fh.close()
																	#  Was the underactuated gripper successful?
							underactuated_fh = open(os.path.join(underactuated_path, underactuated_files[i]), 'r')
							underactuated_lines = underactuated_fh.readlines()
							j = 0
							while j < len(underactuated_lines):
								underactuated_arr = underactuated_lines[j].strip().split('\t')
								if int(underactuated_arr[0]) == trial_index:
									underactuated_success = underactuated_arr[1] == '1'
									break
								j += 1
							underactuated_fh.close()
																	#  Save estimator's prediction and trial outcomes.
							data[estimator_name][ (dataset_name, int(arr[0]), int(arr[1]), int(object_id)) ][ 'estimate' ] = fh_npz['T_est'][ int(arr[2]) ]
							data[estimator_name][ (dataset_name, int(arr[0]), int(arr[1]), int(object_id)) ][ 'success' ]  = (parallel_success, underactuated_success)

							num_trials += 1

				fh_txt.close()

				if params['verbose']:
					print('    '+dataset_name+', object ' + npz_files[i].split('-')[1] + ':\t' + str(num_trials) + ' trials for parallel and underactuated.')

	timestamp_str = time.strftime('%l:%M%p %Z on %b %d, %Y')
	for estimator in params['estimators']:
		npz = []
		npz_index = 0

		fh = open('dataset-'+estimator+'.txt', 'w')
		fh.write('#  Index into pose-estimate data by "'+estimator+'".\n')
		fh.write('#  CREATED ' + timestamp_str + ':\n')
		fh.write('#    python3 ' + ' '.join(sys.argv) + '\n')
		fh.write('#  ESTIMATOR:' + '\n')
		fh.write('#    '+estimator+ '\n')
		fh.write('#  Made from the following DATASETS:' + '\n')
		fh.write('#    {' + ', '.join(params['datasets']) + '}\n')
		fh.write('#\n')
		fh.write('#  Indices refer to a single NumPy file, "dataset-'+estimator+'.npz", that contains all pose estimates as 4 x 4 rigid transformations.\n')
		fh.write('#\n')
		fh.write('#  LINE FORMAT' + '\n')
		fh.write('#  Dataset <t> Scene <t> Frame <t> Object-ID <t> Visibility <t> Idx."dataset-'+estimator+'.npz" <t> Par.Success('+estimator+') <t> Und.Success('+estimator+')\n')

		print('>>> ' + estimator + ': "dataset-'+estimator+'.txt" & "dataset-'+estimator+'.npz"')

		for kv in sorted([x for x in data[estimator].items()], key=lambda x: x[0]):
			key = kv[0]												#  (dataset, scene, frame, object)
			val = kv[1]												#  Dictionary with 'visibility', 'estimate', 'success'

			fh.write('\t'.join([key[0], str(key[1]), str(key[2]), str(key[3]), str(val['visibility'])]) + '\t')
			fh.write(str(npz_index) + '\t')

			T = val['estimate']										#  Add this pose estimate to the wad.
			if npz_index == 0:
				npz = T
			elif npz_index == 1:
				npz = np.stack([npz, T], axis=0)
			else:
				npz = np.concatenate((npz, T[None]), axis=0)

			if val['success'][0]:									#  Parallel gripper success.
				fh.write('1\t')
			else:
				fh.write('0\t')

			if val['success'][1]:									#  Underactuated gripper success.
				fh.write('1\n')
			else:
				fh.write('0\n')

			npz_index += 1

		np.savez('dataset-'+estimator, npz)
		fh.close()

		print('    Total ' + str(npz.shape[0]) + ' pose estimates.')

	return

def get_command_line_params():
	params = {}

	params['estimators'] = []
	params['datasets'] = []											#  Datasets to include.

	params['estimator-weight-set'] = {('DOPE', 'YCBV'):'net_meat_original_14', \
	                                  ('EPOS', 'YCBV'):'cvpr20-f256', \
	                                  ('EPOS', 'LMO'):'bop20-f64', \
	                                  ('GDRNPP', 'YCBV'):'ycbvSO', \
	                                  ('GDRNPP', 'LMO'):'pbr'}

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-est', '-d', '-dat', '-data', '-dataset', '-v', '-?', '-help', '--help']
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
				if argtarget == '-est':
					params['estimators'].append( argval )
				elif argtarget == '-d' or argtarget == '-dat' or argtarget == '-data' or argtarget == '-dataset':
					params['datasets'].append( argval )

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('This is the first step.')
	print('Collect pose-estimate and grasping-trial data per 6-DoF estimator.')
	print('For each given estimator, this script produces two files:')
	print('  - "dataset-<estimator>.txt" contains pose-estimate conditions, data, and indices into the NumPy file.')
	print('  - "dataset-<estimator>.npz" contains each pose estimate.')
	print('')
	print('Usage:  python3 collect_pose_data.py <parameters, preceded by flags>')
	print(' e.g.:  python3 collect_pose_data.py -est EPOS -est GDRNPP -est ZebraPose -dat YCBV -dat LMO -v')
	print('')
	print('Flags:  -est  MUST HAVE AT LEAST ONE: Following is the name of a 6-DoF pose estimator.')
	print('        -dat  MUST HAVE AT LEAST ONE: Include the following dataset when collecting pose estimates and grasping trials.')
	print('        -v    Enable verbosity')
	print('        -?    Display this message')
	return

if __name__ == '__main__':
	main()
