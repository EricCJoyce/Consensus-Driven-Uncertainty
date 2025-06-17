import cv2
import json
import numpy as np
import os

def main():
	render_video = False
	verbose      = True

	fair_game    = ['obj' + str(x).rjust(2, '0') for x in range(1, 4)]
	root_path    = '/media/eric/Hoboken/Projects/Uncertainty for 6DoF Pose Est/Datasets/BOP/'
	dataset_path = 'tudl/'
	set_path     = 'test/'

	fh_err = open('tudl_pose_error.txt', 'w')
	fh_err.write('#  Pose errors for ZebraPose on TUDL.\n')
	fh_err.write('#  Scene-number  <t>  Keyframe-number  <t>  Object  <t>  Visibility-ratio  <t>  Score  <t>  Gr.Tr.Rotation(space-sep, 9)  <t>  Gr.Tr.Translation(space-sep, 3)  <t>  Est.Rotation(space-sep, 9)  <t>  Est.Translation(space-sep, 3)  <t>  Rotation-error(degrees)  <t>  Translation-error(mm)\n')
																	#  TLESS requires that we consider symmetries to evaluate fairly.
	fh = open(root_path + dataset_path + 'models/models_info.json', 'r')
	json_models = json.load(fh)										#  Keys are integers as strings: obj01 ==> "1"; obj02 ==> "2"; obj03 ==> "3"
	fh.close()														#  Check if 'symmetries_continuous' in json_models[obj_str]
																	#     or if 'symmetries_discrete' in json_models[obj_str].
	for scene in [str(x).rjust(6, '0') for x in range(1, 4)]:		#  For every scene in the test set...
		if verbose:
			print('>>> Scene ' + scene)

		fh_gt = open(root_path + dataset_path + set_path + scene + '/scene_gt.json', 'r')
		json_gt = json.load(fh_gt)									#  This JSON contains {R_gt, t_gt, class_id}
		fh_gt.close()												#  (named cam_R_m2c, cam_t_m2c, obj_id) per object per frame.

		fh_gt_info = open(root_path + dataset_path + set_path + scene + '/scene_gt_info.json', 'r')
		json_gt_info = json.load(fh_gt_info)						#  This JSON contains {bbox_obj, bbox_visib, px_count_all,
		fh_gt_info.close()											#  px_count_valid, px_count_visib, visib_fract} per object per frame.

		keyframes = json_gt.keys()									#  Only keyframes contain ground-truth poses.

		for keyframe in keyframes:									#  NOTE for TLESS: We CANNOT assume that there is only one instance of any present object!
			if verbose:
				print('    Frame ' + keyframe)

			#########################################################  Collect all information about this frame.
			gt_objects_in_frame = []								#  Build a list of objects that are truly present.
																	#  Each in list: {key: 'class-name' ==> val: string,
																	#                 key: 'R'          ==> val: R_gt as list of 9,
																	#                 key: 't'          ==> val: t_gt as list of 3,
																	#                 key: 'bbox'       ==> val: [(up-left X, up-left Y), (low-right X, low-right Y)]
																	#                                            in pixel coords,
			for i in range(0, len(json_gt[keyframe])):				#                 key: 'visfrac'    ==> val: visibility fraction}
				gt_obj = json_gt[keyframe][i]
				gt_info = json_gt_info[keyframe][i]
				gt_objects_in_frame.append( {} )
				gt_objects_in_frame[-1]['class-name'] = lookup_class_name(int(gt_obj['obj_id']))
																	#  Not all ground-truth rotation matrices are actually rotation matrices!
				gt_objects_in_frame[-1]['R']          = np.array(gt_obj['cam_R_m2c']).reshape((3, 3))
																	#  Perform SVD, drop the eigenvalues in Sigma (because they should be identity anyway.)
				U, _, V = np.linalg.svd(gt_objects_in_frame[-1]['R'])
				gt_objects_in_frame[-1]['R']          = np.dot(U, V)#  The trusted matrix is the dot product of U and Vh.
																	#  For now, keep it as a flattened list of nine.
				gt_objects_in_frame[-1]['R']          = list(gt_objects_in_frame[-1]['R'].reshape((9, )))
				gt_objects_in_frame[-1]['t']          = gt_obj['cam_t_m2c']
				gt_objects_in_frame[-1]['bbox']       = ( (gt_info['bbox_obj'][0], gt_info['bbox_obj'][1]), \
				                                          (gt_info['bbox_obj'][0] + gt_info['bbox_obj'][2], gt_info['bbox_obj'][1] + gt_info['bbox_obj'][3]))
				gt_objects_in_frame[-1]['visfrac']    = gt_info['visib_fract']

			predicted_objects_in_frame = []							#  Build a list of detected objects.
																	#  Each in list: {key: 'class-name' ==> val: string,
																	#                 key: 'score'      ==> val: float,
																	#                 key: 'R'          ==> val: R_pred as list of 9,
																	#                 key: 't'          ==> val: t_pred as list of 3,
																	#                 key: 'bbox'       ==> val: [(up-left X, up-left Y), (low-right X, low-right Y)]
			for obj in fair_game:									#                                            in pixel coords}
				fh = open('evaluation_output/pose_result_bop/tudl/' + dataset_path[:-1] + '_' + obj + '.csv', 'r')
				lines_csv = fh.readlines()[1:]						#  Skip header.
				fh.close()

				fh = open('evaluation_output/pose_result_bop/tudl/' + dataset_path[:-1] + '_' + obj + '.bbox', 'r')
				lines_bbox = fh.readlines()
				fh.close()

				assert len(lines_csv) == len(lines_bbox), 'ERROR: "'+dataset_path[:-1]+'_'+obj+'.csv" and "'+dataset_path[:-1]+'_'+obj+'.bbox'+'" are out of sync.'

				for i in range(0, len(lines_bbox)):
					arr_csv = lines_csv[i].strip().split(',')
					arr_bbox = lines_bbox[i].strip().split()

					scene_id     = arr_csv[0]
					frame_number = arr_csv[1]
					obj_id       = int(arr_csv[2])
					if scene_id == scene and frame_number == keyframe.rjust(6, '0') and lookup_class_name(obj_id) not in predicted_objects_in_frame:
						predicted_objects_in_frame.append( {} )
						predicted_objects_in_frame[-1]['class-name'] = lookup_class_name(obj_id)
						predicted_objects_in_frame[-1]['score']      = float(arr_csv[3])
						predicted_objects_in_frame[-1]['R']          = [float(x) for x in arr_csv[4].split()]
						predicted_objects_in_frame[-1]['t']          = [float(x) for x in arr_csv[5].split()]
						predicted_objects_in_frame[-1]['bbox']       = ((int(arr_bbox[0]), int(arr_bbox[1])), \
						                                                (int(arr_bbox[0]) + int(arr_bbox[2]), int(arr_bbox[1]) + int(arr_bbox[3])))

			if verbose:
				print('        Gr.Tr. [' + ' '.join([gt_obj['class-name'] for gt_obj in gt_objects_in_frame]) + ']')
				print('        Pred.  [' + ' '.join([gt_obj['class-name'] for gt_obj in gt_objects_in_frame]) + ']')

			#########################################################  Match predictions to ground truths.
			for pred_obj in predicted_objects_in_frame:
				best_iou = 0.0
				best_match_index = None
				for i in range(0, len(gt_objects_in_frame)):
					if pred_obj['class-name'] == gt_objects_in_frame[i]['class-name']:
						iou = intersection_over_union(pred_obj['bbox'], gt_objects_in_frame[i]['bbox'])
						if iou > best_iou:
							best_iou = iou
							best_match_index = i

				if best_match_index is not None:					#  Matched prediction 'pred_obj' to gt_objects_in_frame[best_match_index].
					obj_id = str(lookup_class_id(pred_obj['class-name']))

					R_gt = gt_objects_in_frame[best_match_index]['R']
					t_gt = gt_objects_in_frame[best_match_index]['t']

					R_pred = pred_obj['R']
					t_pred = pred_obj['t']
																	#  DISCRETE SYMMETRY present: try all transforms; accept the one that minimzes error.
					if 'symmetries_discrete' in json_models[obj_id]:
						R_pred_arr = np.array(R_pred).reshape((3, 3))
						R_gt_arr = np.array(R_gt).reshape((3, 3))
																	#  Error for prediction * identity
						min_err_R = rotation_error(R_pred_arr, R_gt_arr, False)
																	#  Now try all transformations.
						for disc_symm in json_models[str(obj_id)]['symmetries_discrete']:
							R_symm = np.array(disc_symm).reshape((4, 4))[:3, :3]
							R = R_pred_arr.dot(R_symm)
							err_R = rotation_error(R, R_gt_arr, False)
							if err_R < min_err_R:
								min_err_R = err_R					#  Update the minimum.
								R_pred = list(R.reshape((9, )))		#  Overwrite the flattened 9-float.

						fh_err.write(scene + '\t' + keyframe.rjust(6, '0') + '\t' + pred_obj['class-name'] + '\t')
						fh_err.write(str(gt_objects_in_frame[best_match_index]['visfrac']) + '\t' + str(pred_obj['score']) + '\t')
																	#  Write ground truth rotation and translation.
						fh_err.write(' '.join([str(x) for x in R_gt]) + '\t')
						fh_err.write(' '.join([str(x) for x in t_gt]) + '\t')

						fh_err.write(' '.join([str(x) for x in R_pred]) + '\t')
						fh_err.write(' '.join([str(x) for x in t_pred]) + '\t')

						err_R = rotation_error(np.array(R_pred).reshape((3, 3)), np.array(R_gt).reshape((3, 3)), False)
						err_t = translation_error(np.array(t_pred).reshape((3, 1)), np.array(t_gt).reshape((3, 1)))

						fh_err.write(str(err_R) + '\t')
						fh_err.write(str(err_t) + '\n')
																	#  CONTINUOUS SYMMETRY present: roll is irrelevant; error equals divergence from axis.
					elif 'symmetries_continuous' in json_models[obj_id]:
						fh_err.write(scene + '\t' + keyframe.rjust(6, '0') + '\t' + pred_obj['class-name'] + '\t')
						fh_err.write(str(gt_objects_in_frame[best_match_index]['visfrac']) + '\t' + str(pred_obj['score']) + '\t')
																	#  Write ground truth rotation and translation.
						fh_err.write(' '.join([str(x) for x in R_gt]) + '\t')
						fh_err.write(' '.join([str(x) for x in t_gt]) + '\t')

						fh_err.write(' '.join([str(x) for x in R_pred]) + '\t')
						fh_err.write(' '.join([str(x) for x in t_pred]) + '\t')

						a = np.array(json_models[obj_id]['symmetries_continuous'][0]['axis'])
						R_pred_arr = np.array(R_pred).reshape((3, 3))
						R_gt_arr = np.array(R_gt).reshape((3, 3))

						a_prime = R_pred_arr.dot(a)
						a_doubleprime = R_gt_arr.T.dot(a_prime)
						err_R = np.arccos(np.dot(a, a_doubleprime))

						err_t = translation_error(np.array(t_pred).reshape((3, 1)), np.array(t_gt).reshape((3, 1)))

						fh_err.write(str(err_R) + '\t')
						fh_err.write(str(err_t) + '\n')
					else:											#  Else, no symmetries present in this object.
						fh_err.write(scene + '\t' + keyframe.rjust(6, '0') + '\t' + pred_obj['class-name'] + '\t')
						fh_err.write(str(gt_objects_in_frame[best_match_index]['visfrac']) + '\t' + str(pred_obj['score']) + '\t')
																	#  Write ground truth rotation and translation.
						fh_err.write(' '.join([str(x) for x in R_gt]) + '\t')
						fh_err.write(' '.join([str(x) for x in t_gt]) + '\t')

						fh_err.write(' '.join([str(x) for x in R_pred]) + '\t')
						fh_err.write(' '.join([str(x) for x in t_pred]) + '\t')

						err_R = rotation_error(np.array(R_pred).reshape((3, 3)), np.array(R_gt).reshape((3, 3)), False)
						err_t = translation_error(np.array(t_pred).reshape((3, 1)), np.array(t_gt).reshape((3, 1)))

						fh_err.write(str(err_R) + '\t')
						fh_err.write(str(err_t) + '\n')

					del gt_objects_in_frame[best_match_index]		#  Take this matched element out of the running.

				else:												#  The predicted object matches nothing in the ground truth: hallucination.
					R_pred = pred_obj['R']
					t_pred = pred_obj['t']

					fh_err.write(scene + '\t' + keyframe.rjust(6, '0') + '\t' + pred_obj['class-name'] + '\t')
					fh_err.write('*\t' + str(pred_obj['score']) + '\t')

					fh_err.write('*\t')								#  In this case, there ARE NO ground-truth rotation and translation.
					fh_err.write('*\t')

					fh_err.write(' '.join([str(x) for x in R_pred]) + '\t')
					fh_err.write(' '.join([str(x) for x in t_pred]) + '\t')

					fh_err.write('*\t')
					fh_err.write('*\n')

			#########################################################  Now consider ground-truths that ZebraPose missed, if any.
			for gt_obj in gt_objects_in_frame:
				R_gt = gt_obj['R']
				t_gt = gt_obj['t']

				fh_err.write(scene + '\t' + keyframe.rjust(6, '0') + '\t*\t')
				fh_err.write(str(gt_obj['visfrac']) + '\t*\t')

				fh_err.write(' '.join([str(x) for x in R_gt]) + '\t')
				fh_err.write(' '.join([str(x) for x in R_gt]) + '\t')

				fh_err.write('*\t')									#  In this case, there are no predicted rotation and translation.
				fh_err.write('*\t')

				fh_err.write('*\t')
				fh_err.write('*\n')

	fh_err.close()

	return

#  Receives two ROTATION MATRICES.
#  Returns radians or degrees
def rotation_error(R_hat, R_gt, radians=True):
	epsilon = 0.000001

	cos_theta = (np.trace(np.dot(R_hat, R_gt.T)) - 1.0) * 0.5

	if cos_theta > 1.0:
		assert cos_theta - 1.0 < epsilon, 'ERROR in rotation error computation: difference '+str(cos_theta - 1.0)+' greater than epsilon.'
		cos_theta = 1.0
	elif cos_theta < -1.0:
		assert cos_theta + 1.0 < epsilon, 'ERROR in rotation error computation: sum '+str(cos_theta + 1.0)+' greater than epsilon.'
		cos_theta = -1.0

	if radians:
		return np.arccos(cos_theta)
	return np.arccos(cos_theta) * (180.0 / np.pi)

def translation_error(t_hat, t_gt):
	return np.linalg.norm(t_gt - t_hat)

#  box = ((x, y), (x, y))
#  scale = (w, h)
def intersection_over_union(boxA, boxB, scale=(1.0, 1.0)):
	A = ((boxA[0][0] * scale[0], boxA[0][1] * scale[1]), (boxA[1][0] * scale[0], boxA[1][1] * scale[1]))
	B = ((boxB[0][0] * scale[0], boxB[0][1] * scale[1]), (boxB[1][0] * scale[0], boxB[1][1] * scale[1]))

	xA = max(A[0][0], B[0][0])
	yA = max(A[0][1], B[0][1])
	xB = min(A[1][0], B[1][0])
	yB = min(A[1][1], B[1][1])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (A[1][0] - A[0][0] + 1) * (A[1][1] - A[0][1] + 1)
	boxBArea = (B[1][0] - B[0][0] + 1) * (B[1][1] - B[0][1] + 1)

	if boxAArea + boxBArea - interArea == 0:
		return 0.0

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def lookup_class_name(class_id):
	return 'obj' + str(class_id).rjust(2, '0')

def lookup_class_id(class_name):
	return int(class_name[3:])

if __name__ == '__main__':
	main()
