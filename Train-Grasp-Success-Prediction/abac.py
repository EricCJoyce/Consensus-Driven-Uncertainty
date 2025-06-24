import numpy as np
import os
import sys
import time

class PoseDifferenceMode:
	TRANS_ERR_SCALAR_ROT_ERR_SCALAR           = 0					#  Store translation error as a scalar and rotation error as a scalar.
	TRANS_ERR_3VEC_ROT_ERR_3VEC               = 1					#  Store translation error as a SIGNED 3-vec(x, y, z) and rotation error as a SIGNED 3-vec(axis-angle minus axis).
	TRANS_ERR_SCALAR_ROT_ERR_3VEC             = 2					#  Store translation error as a scalar and rotation error as a SIGNED 3-vec(axis-angle minus axis).
	TRANS_ERR_3VEC_ROT_ERR_SCALAR             = 3					#  Store translation error as a SIGNED 3-vec(x, y, z) and rotation error as a scalar.
	AVERAGE_DISTANCE_DISTINGUISHABLE          = 4					#  Store the average distance (single scalar) between corresponding points in each pose.
	WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE = 5					#  Store the (weighted) average distance (single scalar) between corresponding points in each pose.

def main():
	params = get_command_line_params()								#  Collect parameters.
	if params['principal'] is None or len(params['support']) == 0 or params['helpme']:
		usage()
		return

	principal_supp_combos = [params['principal'] + '-' + x for x in params['support']]

	timestamp_str = time.strftime('%l:%M%p %Z on %b %d, %Y')
	dataset_source_armed = False									#  Trigger indicating whether we have begun reading source datasets.
	fh = open('dataset-'+params['principal']+'.txt', 'r')
	for line in fh.readlines():
		if '#  Made from the following DATASETS:' in line:
			dataset_source_armed = True
		elif dataset_source_armed:
			datasets_string = line[line.index('{')+1:line.index('}')]
			dataset_source_armed = False
	fh.close()

	npz_index = 0													#  Prepare the index into the file "dataset-<principal>-<mode>.npz" to be created.

	if params['mode'] == PoseDifferenceMode.TRANS_ERR_SCALAR_ROT_ERR_SCALAR:
		fh = open('dataset-'+params['principal']+'-t1r1.txt', 'w')
	elif params['mode'] == PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC:
		fh = open('dataset-'+params['principal']+'-t3r3.txt', 'w')
	elif params['mode'] == PoseDifferenceMode.TRANS_ERR_SCALAR_ROT_ERR_3VEC:
		fh = open('dataset-'+params['principal']+'-t1r3.txt', 'w')
	elif params['mode'] == PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_SCALAR:
		fh = open('dataset-'+params['principal']+'-t3r1.txt', 'w')
	elif params['mode'] == PoseDifferenceMode.AVERAGE_DISTANCE_DISTINGUISHABLE:
		fh = open('dataset-'+params['principal']+'-add.txt', 'w')
	elif params['mode'] == PoseDifferenceMode.WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE:
		fh = open('dataset-'+params['principal']+'-wadd.txt', 'w')

	fh.write('#  Dataset for an ensemble grasp-success predictor (including both grippers), given that "'+params['principal']+'" is the PRINCIPAL pose-estimator.\n')
	fh.write('#  Created ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
	fh.write('#    python3 '+' '.join(sys.argv)+'\n')
	fh.write('#  PRINCIPAL ESTIMATOR:' + '\n')
	fh.write('#    '+params['principal']+ '\n')
	fh.write('#  SUPPORT ESTIMATOR(S):' + '\n')
	fh.write('#    {' + ', '.join(params['support']) + '}\n')
	fh.write('#  DATASET(S):' + '\n')
	fh.write('#    {' + datasets_string + '}\n')
	fh.write('#  FORMAT:\n')
	fh.write('#    A-B A-C\n')
	fh.write('#  MODE:\n')
	if params['mode'] == PoseDifferenceMode.AVERAGE_DISTANCE_DISTINGUISHABLE:
		fh.write('#    '+params['mode-str']+' (meters)\n')
	if params['mode'] == PoseDifferenceMode.WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE:
		fh.write('#    '+params['mode-str']+'\n')
	else:
		fh.write('#    '+params['mode-str']+' (meters, radians)\n')
	fh.write('#\n')
	fh.write('#  Indices refer to a single NumPy file, "dataset-'+params['principal']+'-'+params['mode-str']+'.npz", that contains all pose-estimate differences.\n')
	fh.write('#\n')
	fh.write('#  LINE FORMAT' + '\n')
	fh.write('#  Dataset <t> Scene <t> Frame <t> Object-ID <t> Visibility <t> ')
	fh.write(' <t> '.join(['Idx.diff.('+x+')' for x in principal_supp_combos]))
	fh.write(' <t> Par.Success('+params['principal']+')')
	fh.write(' <t> Und.Success('+params['principal']+')\n')

	#################################################################  Gather principal-supporter pose-estimate discrepancies.

	fh_principal_idx = open('dataset-'+params['principal']+'.txt', 'r')
	fh_principal_npz = np.load('dataset-'+params['principal']+'.npz')
	for line in fh_principal_idx.readlines():
		if line[0] != '#':
			arr = line.strip().split('\t')
			dataset      = arr[0]
			scene        = int(arr[1])
			frame        = int(arr[2])
			object_id    = int(arr[3])
			visibility   = float(arr[4])
			estimate_idx = int(arr[5])
			T            = fh_principal_npz['arr_0'][ estimate_idx ]
			parallel_success = arr[6] == '1'
			underactuated_success = arr[7] == '1'

			support_ests = [None for x in params['support']]		#  We can only save records for which the principal and all the supporters had estimates.

			support_ctr = 0
			for supporter in params['support']:
				fh_support_idx = open('dataset-'+supporter+'.txt', 'r')
				fh_support_npz = np.load('dataset-'+supporter+'.npz')

				for support_line in fh_support_idx.readlines():
					if support_line[0] != '#':
						support_arr = support_line.strip().split('\t')
						support_dataset      = support_arr[0]
						support_scene        = int(support_arr[1])
						support_frame        = int(support_arr[2])
						support_object_id    = int(support_arr[3])
																	#  Supporting estimator has an estimate for the same (dataset, scene, frame, object)!
						if support_dataset == dataset and support_scene     == scene and \
						   support_frame   == frame   and support_object_id == object_id:
							support_ests[support_ctr] = fh_support_npz['arr_0'][ int(support_arr[5]) ]

				fh_support_idx.close()
				support_ctr += 1

			if len([x for x in support_ests if x is None]) == 0:	#  All supporters have a pose estimate: store for training.
				if params['verbose']:
					print(params['principal']+' record found for '+dataset+' sc. '+str(scene)+' fr. '+str(frame)+', obj.'+str(object_id)+': {'+', '.join(params['support'])+'}')

				fh.write(dataset+'\t'+str(scene)+'\t'+str(frame)+'\t'+str(object_id)+'\t'+str(visibility)+'\t')

				for T_support in support_ests:
					if params['mode'] == PoseDifferenceMode.TRANS_ERR_SCALAR_ROT_ERR_SCALAR:
						D_t, D_r = t1r1(T, T_support)
						D = np.array([D_t, D_r])					#  Form a 1-by-2.
						if params['verbose']:
							print('\t', [D_t, D_r])
					elif params['mode'] == PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC:
						D_t, D_r = t3r3(T, T_support)
						D = np.hstack([D_t, D_r])					#  Form a 3-by-2.
						if params['verbose']:
							print('\t', list(D_t.reshape((3, ))) + list(D_r.reshape((3, ))))
					elif params['mode'] == PoseDifferenceMode.TRANS_ERR_SCALAR_ROT_ERR_3VEC:
						D_t, D_r = t1r3(T, T_support)
						D = np.hstack([np.array([[D_t]]), D_r.T])	#  Form a 1-by-4.
						if params['verbose']:
							print('\t', [D_t] + list(D_r.reshape((3, ))))
					elif params['mode'] == PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_SCALAR:
						D_t, D_r = t3r1(T, T_support)
						D = np.hstack([D_t.T, np.array([[D_r]])])	#  Form a 1-by-4.
						if params['verbose']:
							print('\t', list(D_t.reshape((3, ))) + [D_r])
					elif params['mode'] == PoseDifferenceMode.AVERAGE_DISTANCE_DISTINGUISHABLE:
																	#  Scalar.
						D = ADD(T, T_support, dataset, object_id, params)
						if params['verbose']:
							print('\t', D)
					elif params['mode'] == PoseDifferenceMode.WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE:
																	#  Scalar.
						D = wADD(T, T_support, dataset, object_id, params)
						if params['verbose']:
							print('\t', D)

					fh.write(str(npz_index)+'\t')

					if npz_index == 0:
						npz = D
					elif npz_index == 1:
						npz = np.stack([npz, D], axis=0)
					else:
						npz = np.concatenate((npz, D[None]), axis=0)

					npz_index += 1

				if parallel_success:
					fh.write('1\t')
				else:
					fh.write('0\t')
				if underactuated_success:
					fh.write('1\n')
				else:
					fh.write('0\n')

	fh_principal_idx.close()

	np.savez('dataset-'+params['principal']+'-'+params['mode-str'], npz)

	fh.close()

	return

def t1r1(T_0, T_1, radians=True):
	t_0 = T_0[:3, 3].reshape((3, 1))
	t_1 = T_1[:3, 3].reshape((3, 1))
	R_0 = T_0[:3, :3]
	R_1 = T_1[:3, :3]
	return translation_scalar_difference(t_0, t_1), rotation_scalar_difference(R_0, R_1, radians)

def t3r3(T_0, T_1, radians=True):
	t_0 = T_0[:3, 3].reshape((3, 1))
	t_1 = T_1[:3, 3].reshape((3, 1))
	R_0 = T_0[:3, :3]
	R_1 = T_1[:3, :3]
	return translation_3vec_difference(t_0, t_1), rotation_3vec_difference(R_0, R_1)

def t1r3(T_0, T_1, radians=True):
	t_0 = T_0[:3, 3].reshape((3, 1))
	t_1 = T_1[:3, 3].reshape((3, 1))
	R_0 = T_0[:3, :3]
	R_1 = T_1[:3, :3]
	return translation_scalar_difference(t_0, t_1), rotation_3vec_difference(R_0, R_1)

def t3r1(T_0, T_1, radians=True):
	t_0 = T_0[:3, 3].reshape((3, 1))
	t_1 = T_1[:3, 3].reshape((3, 1))
	R_0 = T_0[:3, :3]
	R_1 = T_1[:3, :3]
	return translation_3vec_difference(t_0, t_1), rotation_scalar_difference(R_0, R_1, radians)

def translation_scalar_difference(t_0, t_1):
	return np.linalg.norm(t_0 - t_1)

#  20aug24: SIGNED!
def translation_3vec_difference(t_0, t_1):
	return t_0 - t_1

def rotation_scalar_difference(R_0, R_1, radians=True):
	cos_theta = (np.trace(np.dot(R_0, R_1.T)) - 1.0) * 0.5
	if radians:
		return np.arccos(cos_theta)
	return np.arccos(cos_theta) * (180.0 / np.pi)

#  20aug24: SIGNED!
def rotation_3vec_difference(R_0, R_1):
	e_0 = rotationMatrixToEulerAngles(R_0)
	e_1 = rotationMatrixToEulerAngles(R_1)
	return e_0 - e_1

#  Return Euler angles as radians in Z, Y, X
#  Checks out: e.g.
#  [ -0.47917222  -0.63630031   0.6045791  ]                                                       [ -1.82639749 ]
#  [ -0.5364678   -0.33285346  -0.77550685 ] ==> [ x: -1.8263975, y: -0.7679909, z: -2.299841 ] == [ -0.76799092 ]
#  [  0.6946915   -0.69593856  -0.1818605  ]                                                       [ -2.29984095 ]
def rotationMatrixToEulerAngles(R):
	sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
	singular = sy < 1e-6
	if not singular:
		x = np.arctan2( R[2, 1], R[2, 2])
		y = np.arctan2(-R[2, 0], sy)
		z = np.arctan2( R[1, 0], R[0, 0])
	else:
		x = np.arctan2(-R[1, 2], R[1, 1])
		y = np.arctan2(-R[2, 0], sy)
		z = 0.0
	return np.array([x, y, z]).reshape((3, 1))

#  Return Euler angles as radians in ZYX as X, Y, Z
def rotMat2YawPitchRoll(R):
	y = np.arctan2( R[1, 0], R[0, 0])								#  z
	p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))	#  y
	r = np.arctan2( R[2, 1], R[2, 2])								#  x
	return np.array([y, p, r]).reshape((3, 1))

#  ZXZ: https://en.wikipedia.org/wiki/Euler_angles
def rotMat2RollPitchYaw(R):
	psi   = -np.arctan2(R[0, 2], R[1, 2])
	phi   =  np.arctan2(R[2, 0], R[2, 1])
	theta =  np.arccos(R[2, 2])
	return np.array([phi, theta, psi]).reshape((3, 1))

#  Compute the average distance between corresponding points in pose T and pose T_support.
#  This is object-dependent.
def ADD(T, T_support, dataset, object_id, params):
	model_path = params['BOP-path-stem']+dataset.lower()+'/models_eval_ascii/obj_'+str(object_id).rjust(6, '0')+'.ply'
	fh_ply = open(model_path, 'r')
	ply_lines = fh_ply.readlines()
	fh_ply.close()

	i = 0
	while i < len(ply_lines):										#  Retrieve count of vertices from the header.
		if 'element vertex' in ply_lines[i]:
			arr = ply_lines[i].strip().split()
			num_vertices = int(arr[-1])
		elif 'end_header' in ply_lines[i]:
			i += 1
			break
		i += 1

	V = {}															#  Build set of vertices.
	j = 0
	while j < num_vertices:
		arr = ply_lines[i + j].strip().split()
																	#  Make homogeneous coordinates for multiplication with 4x4.
																	#  CONVERT FROM MILLIMETERS(BOP's units) TO METERS(Stevens' units).
		V[j] = (float(arr[0]) * 0.001, float(arr[1]) * 0.001, float(arr[2]) * 0.001, 1.0)
		j += 1
	i += num_vertices

	D = []															#  Accumulate corresponding differences.
	for i in range(0, len(V)):
																	#  Compute point in principal's pose estimate.
		x_T         = np.dot(T,         np.array(V[i]).reshape((4, 1)))
																	#  Compute point in supporter's pose estimate.
		x_T_support = np.dot(T_support, np.array(V[i]).reshape((4, 1)))
		D.append( np.linalg.norm(x_T_support - x_T, 2) )			#  Add to running list of Euclidean distances.

	return np.mean(D)

#  Compute the average distance between corresponding points in pose T and pose T_support--WEIGHTED by the areas of the triangles to which each point contributes.
#  This is object-dependent.
def wADD(T, T_support, dataset, object_id, params):
	model_path = params['BOP-path-stem']+dataset.lower()+'/models_eval_ascii/obj_'+str(object_id).rjust(6, '0')+'.ply'
	fh_ply = open(model_path, 'r')
	ply_lines = fh_ply.readlines()
	fh_ply.close()

	i = 0
	while i < len(ply_lines):										#  Retrieve counts of vertices and faces from the header.
		if 'element vertex' in ply_lines[i]:
			arr = ply_lines[i].strip().split()
			num_vertices = int(arr[-1])
		elif 'element face' in ply_lines[i]:
			arr = ply_lines[i].strip().split()
			num_faces = int(arr[-1])
		elif 'end_header' in ply_lines[i]:
			i += 1
			break
		i += 1

	V = {}															#  Build set of vertices.
	j = 0
	while j < num_vertices:
		arr = ply_lines[i + j].strip().split()
																	#  CONVERT FROM MILLIMETERS(BOP's units) TO METERS(Stevens' units).
		V[j] = (float(arr[0]) * 0.001, float(arr[1]) * 0.001, float(arr[2]) * 0.001)
		j += 1
	i += num_vertices

	F = {}															#  Build set of faces.
	j = 0
	while j < num_faces:
		arr = ply_lines[i + j].strip().split()
																	#  Vector from [1] to [2] (PR).
		pr = np.array(V[ int(arr[2]) ]).reshape((3, 1)) - np.array(V[ int(arr[1]) ]).reshape((3, 1))
																	#  Vector from [1] to [3] (PQ).
		pq = np.array(V[ int(arr[3]) ]).reshape((3, 1)) - np.array(V[ int(arr[1]) ]).reshape((3, 1))
																	#  Area = 0.5 * norm of cross product of pq and pr.
																	#  Key face by vertex-indices ==> store triangle area.
		F[ (int(arr[1]), int(arr[2]), int(arr[3])) ] = 0.5 * np.linalg.norm(np.cross(pq.reshape((3, )), pr.reshape((3, ))))
		j += 1

	D = []															#  Accumulate corresponding differences.
	for i in range(0, len(V)):
		total_triangle_area = 0.0
		for face_key in F.keys():
			if i in face_key:
				total_triangle_area += F[face_key]
		x = np.array(list(V[i]) + [1.0]).reshape((4, 1))			#  Make homogeneous coordinates for multiplication with 4x4.
		x_T         = np.dot(T, x)									#  Compute point in principal's pose estimate.
		x_T_support = np.dot(T_support, x)							#  Compute point in supporter's pose estimate.
		D.append( np.linalg.norm(x_T_support[:3] - x_T[:3], 2) )	#  Add to running list of Euclidean distances.

	return np.mean(D) * total_triangle_area

def get_command_line_params():
	params = {}

	params['principal'] = None										#  Principal estimator.
	params['support'] = []											#  Support estimators to include.

	params['mode'] = PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC
	params['mode-str'] = 't3r3'
	params['mode-args'] = ['t1r1', 't3r3', 't1r3', 't3r1', 'add', 'wadd']

	params['BOP-path-stem'] = '/media/Hoboken/Projects/Is Image-based Object Pose Estimation Ready to Support Grasping/V1/Datasets/BOP/'

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-principal', '-support', '-mode', '-v', '-?', '-help', '--help']
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
				elif argtarget == '-support':
					params['support'].append( argval )
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
					elif argval.lower() == 'add':
						params['mode'] = PoseDifferenceMode.AVERAGE_DISTANCE_DISTINGUISHABLE
						params['mode-str'] = argval.lower()
					elif argval.lower() == 'wadd':
						params['mode'] = PoseDifferenceMode.WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE
						params['mode-str'] = argval.lower()

	return params

#  Explain usage of this script and its options to the user.
def usage():
	print('This is a second step. This script makes training-ready datasets per PRINCIPAL POSE ESTIMATOR.')
	print('')
	print('NOTE: This script follows an A-B A-C pattern.')
	print('      Meaning that it does not store estimate disagreements between support estimators.')
	print('')
	print('Use pose estimates and grasping trials to build a dataset for an ensemble grasp-success predictor.')
	print('Network input can be:')
	print('  - TRANSLATION difference between the "Principal" estimator and each supporting estimator:')
	print('    - as 1 scalar per supporting estimator.')
	print('    - as 3 scalars (x, y, z) per supporting estimator.')
	print('  - ROTATION difference between the "Principal" estimator and each supporting estimator:')
	print('    - as 1 scalar per supporting estimator.')
	print('    - as 3 scalars (roll, pitch, yaw) per supporting estimator.')
	print('  - Average Distance between Distinguishable points (ADD)')
	print('    - as 1 scalar per supporting estimator')
	print('  - Weighted Average Distance between Distinguishable points (wADD)')
	print('    - as 1 scalar per supporting estimator')
	print('')
	print('Usage:  python3 abac.py <parameters, preceded by flags>')
	print(' e.g.:  python3 abac.py -principal EPOS -support GDRNPP -support ZebraPose -mode t3r3 -v')
	print(' e.g.:  python3 abac.py -principal GDRNPP -support EPOS -support ZebraPose -mode t3r3 -v')
	print(' e.g.:  python3 abac.py -principal ZebraPose -support EPOS -support GDRNPP -mode t3r3 -v')
	print('')
	print('Flags:  -principal  REQUIRED: Following 6-DoF pose estimator is the Principal Estimator.')
	print('        -support    MUST HAVE AT LEAST ONE: Following 6-DoF pose estimator is a supporting estimator.')
	print('        -mode       In {t1r1, t3r3, t1r3, t3r1, add, wadd}. Default is t3r3.')
	print('                    Determines how a pose discrepancy is stored/given to network.')
	print('        -v          Enable verbosity')
	print('        -?          Display this message')
	return

if __name__ == '__main__':
	main()
