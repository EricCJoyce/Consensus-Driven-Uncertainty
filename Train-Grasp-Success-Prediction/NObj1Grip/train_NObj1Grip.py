from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import time
import torch
import torch.nn as nn

from GraspSuccessEstimatorNetwork import GraspSuccessEstimatorNetwork
from Dataset_NObj1Grip import Dataset_NObj1Grip
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import DataLoader

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
	TRANS_ERR_SCALAR_ROT_ERR_SCALAR             = 0					#  Store translation error as a scalar and rotation error as a scalar.
	TRANS_ERR_3VEC_ROT_ERR_3VEC                 = 1					#  Store translation error as a SIGNED 3-vec(x, y, z) and rotation error as a SIGNED 3-vec(axis-angle minus axis).
	TRANS_ERR_SCALAR_ROT_ERR_3VEC               = 2					#  Store translation error as a scalar and rotation error as a SIGNED 3-vec(axis-angle minus axis).
	TRANS_ERR_3VEC_ROT_ERR_SCALAR               = 3					#  Store translation error as a SIGNED 3-vec(x, y, z) and rotation error as a scalar.
	TRANS_3VEC_ROT_3VEC_TRANS_SCALAR_ROT_SCALAR = 4
	AVERAGE_DISTANCE_DISTINGUISHABLE            = 5					#  Store the average distance (single scalar) between corresponding points in each pose.
	WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE   = 6					#  Store the (weighted) average distance (single scalar) between corresponding points in each pose.

class textcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

def main():
	params = get_command_line_params()
	if params['principal-estimator'] is None or params['objects-set-name'] is None or params['helpme']:
		usage()
		return
																	#  Identify mode.
	training_data = Dataset_NObj1Grip( [os.path.join('..', 'DataSplit', params['mode-str'], \
	                                                 '-'.join(['train', params['principal-estimator'], x, params['gripper-str'], params['mode-str']])+'.txt') \
	                                    for x in params['object-strs']] )
	train_dataloader = DataLoader(training_data, batch_size=params['batch-size'], shuffle=True, num_workers=4)
																	#  VERY important that you set SHUFFLE to FALSE for the TEST SET!
																	#  We want to be able to look up which frame a test sample came from.
	test_data = Dataset_NObj1Grip( [os.path.join('..', 'DataSplit', params['mode-str'], \
	                                             '-'.join(['test', params['principal-estimator'], x, params['gripper-str'], params['mode-str']])+'.txt') \
	                                for x in params['object-strs']] )
	test_dataloader = DataLoader(test_data, batch_size=params['batch-size'], shuffle=False, num_workers=4)

	input_size = training_data.input_length							#  Save input size and pass to model constructor.

	available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
	if len(available_gpus) <= 1:
		device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
	elif params['prefer-gpu'] is not None:
		device = ('cuda:' + str(params['prefer-gpu']) if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
	print('\n' + textcolors.HEADER + f"Using device: {device}" + textcolors.ENDC + '\n')

	least_training_loss = float('inf')								#  Initialize least losses.
	least_testing_loss = float('inf')
	best_model = None												#  Update if training loss improves AND test loss improves.
	best_model_test = None											#  Update if test loss improves.
																	#  (Yes, this is unfair, but I want a sense of best model performance.)
	learning_rate = []												#  Be able to plot learning rate and decay.

	if params['resume']:											#  Resuming from a previous checkpoint?
		model_checkpoints = [x for x in os.listdir() if x.startswith('-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str']])) and x.endswith('.pth')]
		model_checkpoints = sorted(model_checkpoints, key=lambda x: int(x.split('-')[2].split('.')[0]))
																	#  Collect all records, too.
		record_checkpoints = [x for x in os.listdir() if x.startswith('-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str']])) and x.endswith('.txt')]
		record_checkpoints = sorted(record_checkpoints, key=lambda x: int(x.split('-')[2].split('.')[0]))

		if len(model_checkpoints) > 0:
			latest_model_name = model_checkpoints[-1]
			latest_completed_epoch = int(latest_model_name.split('-')[-1].split('.')[0])
																	#  Create model.
			model = GraspSuccessEstimatorNetwork(input_size)
			if params['verbose']:
				print(model)
				#print('\nOne-hot vector length = ' + str(len(training_data.objects_declared)) + '\n')
			model.load_state_dict(torch.load(latest_model_name, weights_only=False))
			best_model = latest_model_name
			model = model.to(device)								#  Move model to device.
			print(textcolors.HEADER + 'Resuming training from checkpoint "'+latest_model_name+'"' + textcolors.ENDC + '\n')
			params['initial-epoch'] = latest_completed_epoch + 1

			for record_file in record_checkpoints:					#  Find least recorded losses.
				fh = open(record_file, 'r')
				for line in fh.readlines():
					if line[0] != '#' and 'Training loss:' in line:
						arr = line.strip().split()
						if float(arr[-1]) < least_training_loss:
							least_training_loss = float(arr[-1])
					elif line[0] != '#' and 'Test loss:' in line:
						arr = line.strip().split()
						if float(arr[-1]) < least_testing_loss:
							least_testing_loss = float(arr[-1])
				fh.close()
		else:														#  Create model.
			model = GraspSuccessEstimatorNetwork(input_size).to(device)
			if params['verbose']:
				print(model)
				#print('\nOne-hot vector length = ' + str(len(training_data.objects_declared)) + '\n')
			print(textcolors.WARNING + 'No checkpoints found. Training from scratch.' + textcolors.ENDC)
	else:															#  Create model; move to device.
		model = GraspSuccessEstimatorNetwork(input_size).to(device)
		if params['verbose']:
			print(model)
			#print('\nOne-hot vector length = ' + str(len(training_data.objects_declared)) + '\n')

	#################################################################  Create optimizer.
	if params['optimizer']['type'] == 'SGD':
		optimizer = torch.optim.SGD(model.parameters(), lr=params['learning-rate-schedule']['initial'])
	elif params['optimizer']['type'] == 'Adagrad':
		optimizer = torch.optim.Adagrad(model.parameters(), lr=params['learning-rate-schedule']['initial'])
	elif params['optimizer']['type'] == 'Adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=params['learning-rate-schedule']['initial'])

	#################################################################  Create learning-rate scheduler.
	if params['learning-rate-schedule']['type'] == 'step':
		scheduler = StepLR(optimizer, step_size=params['learning-rate-schedule']['step'], gamma=params['learning-rate-schedule']['gamma'])
	elif params['learning-rate-schedule']['type'] == 'cosine':
		scheduler = CosineAnnealingLR(optimizer, params['epochs'], params['learning-rate-schedule']['min'])
	elif params['learning-rate-schedule']['type'] == 'cosine-warm-restarts':
		scheduler = CosineAnnealingWarmRestarts(optimizer, params['learning-rate-schedule']['iter-to-restart'], \
		                                                   params['learning-rate-schedule']['factor'],          \
		                                                   params['learning-rate-schedule']['min']              )

	if params['verbose']:
		if params['optimizer']['type'] == 'SGD':
			print('- '+textcolors.HEADER+'SGD optimizer('+'{:e}'.format(params['learning-rate-schedule']['initial'])+')'+textcolors.ENDC)
		elif params['optimizer']['type'] == 'Adagrad':
			print('- '+textcolors.HEADER+'Adagrad optimizer('+'{:e}'.format(params['learning-rate-schedule']['initial'])+')'+textcolors.ENDC)
		elif params['optimizer']['type'] == 'Adam':
			print('- '+textcolors.HEADER+'Adam optimizer('+'{:e}'.format(params['learning-rate-schedule']['initial'])+')'+textcolors.ENDC)

		if params['learning-rate-schedule']['type'] == 'step':
			print('- '+textcolors.HEADER+'Step LR scheduler('+str(params['learning-rate-schedule']['step'])+', '+'{:e}'.format(params['learning-rate-schedule']['initial'])+', '+'{:e}'.format(params['learning-rate-schedule']['gamma'])+')'+textcolors.ENDC)
		elif params['learning-rate-schedule']['type'] == 'cosine':
			print('- '+textcolors.HEADER+'Cosine annealing LR scheduler('+'{:e}'.format(params['learning-rate-schedule']['initial'])+', '+'{:e}'.format(params['learning-rate-schedule']['min'])+')'+textcolors.ENDC)
		elif params['learning-rate-schedule']['type'] == 'cosine-warm-restarts':
			print('- '+textcolors.HEADER+'Cosine annealing scheduler with warm restarts('+'{:e}'.format(params['learning-rate-schedule']['initial'])+', '+'{:e}'.format(params['learning-rate-schedule']['min'])+', '+str(params['learning-rate-schedule']['iter-to-restart'])+', '+str(params['learning-rate-schedule']['factor'])+')'+textcolors.ENDC)

	loss_fn = nn.BCELoss()											#  Loss is Binary Cross Entropy.
	if params['verbose']:
		print('- '+textcolors.HEADER+'Binary Cross-Entropy loss'+textcolors.ENDC)

	for t in range(params['initial-epoch'], params['epochs']):		#  Training loop.
		print('Epoch ' + textcolors.WARNING + f"{t}" + textcolors.ENDC + ': ' + \
		      textcolors.OKGREEN + f"{len(train_dataloader.dataset)}" + textcolors.ENDC + \
		      ' train, ' + textcolors.OKBLUE + str(len(test_dataloader.dataset)) + textcolors.ENDC + ' test')
		print('lr: ' + '{:e}'.format(optimizer.param_groups[0]['lr']))

		learning_rate.append(optimizer.param_groups[0]['lr'])		#  Plot learning rate.

		curr_training_loss = train(train_dataloader, model, loss_fn, optimizer, scheduler, device, params)

		if params['report-everything']:
																	#  Identify the current model.
			model_name = '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth'
			curr_testing_loss = test_and_report(test_dataloader, model, loss_fn, device, params, model_name, test_data)
		else:
			curr_testing_loss = test(test_dataloader, model, loss_fn, device, params)
																	#  Save a record of loss regardless of whether we save a model checkpoint.
		fh = open('-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.txt', 'w')
		fh.write('#  Grasp Success Estimator network, epoch ' + str(t).rjust(3, '0') + '\n')
		fh.write('#  PRINCIPAL:   '+params['principal-estimator']+'\n')
		fh.write('#  OBJECT(S):  {'+','.join(params['object-strs'])+'}\n')
		fh.write('#  OBJ.SET:     '+params['objects-set-name']+'\n')
		fh.write('#  GRIPPER(S): {'+params['gripper-str']+'}\n')
		fh.write('#  MODE:        '+params['mode-str']+'\n')
		fh.write('#  ' + time.strftime('%l:%M%p %Z on %b %d, %Y') + '\n')
		fh.write('Training loss:    ' + str(curr_training_loss) + '\n')
		fh.write('Test loss:        ' + str(curr_testing_loss) + '\n')
		fh.close()
																	#  Respond to changes in loss.
		if curr_training_loss < least_training_loss:
			print(f"Training Error: Avg loss: " + textcolors.OKGREEN + f"{curr_training_loss:>8f}" + textcolors.ENDC)
		else:
			print(f"Training Error: Avg loss: " + textcolors.FAIL + f"{curr_training_loss:>8f}" + textcolors.ENDC)

		if curr_testing_loss < least_testing_loss:
			print(f"Test Error:     Avg loss: " + textcolors.OKBLUE + f"{curr_testing_loss:>8f}" + textcolors.ENDC)
		else:
			print(f"Test Error:     Avg loss: " + textcolors.FAIL + f"{curr_testing_loss:>8f}" + textcolors.ENDC)

		if (curr_training_loss < least_training_loss and curr_testing_loss < least_testing_loss):
																	#  Identify the best model.
			best_model = '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth'
																	#  Save a copy of the model.
			if params['save-policy'] == 'improve' or params['save-policy'] == 'all':
				torch.save(model.state_dict(), '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth')
				print(textcolors.HEADER + f"Saved model " + '"' + '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth"' + textcolors.ENDC + '\n')
			else:
				print(textcolors.WARNING + f"Did not save model." + textcolors.ENDC + '\n')
																	#  We may be saving all models...
		elif params['save-policy'] == 'all':						#  but that does not mean every model is the best model.
			torch.save(model.state_dict(), '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth')
			print(textcolors.HEADER + f"Saved model " + '"' + '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth"' + textcolors.ENDC + '\n')
		else:
			print(textcolors.WARNING + f"Did not save model." + textcolors.ENDC + '\n')

		if curr_training_loss < least_training_loss:
			least_training_loss = curr_training_loss				#  Update goalpost.
		if curr_testing_loss < least_testing_loss:
			least_testing_loss = curr_testing_loss					#  Update goalpost.
																	#  Identify best-generalizing model.
			best_model_test = '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], str(t).rjust(3, '0')]) + '.pth'

	print('Done!\n')

	print('Best model is "' + textcolors.OKGREEN + best_model + textcolors.ENDC + '".')
	if params['report-everything']:									#  A confusion matrix and test-set performance already exist. Clone them.
		shutil.copy(best_model[:-4]+'-confusionmatrix.txt', \
		            '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], 'best', 'confusionmatrix']) + '.txt')
		shutil.copy(best_model[:-4]+'-record.txt', \
		            '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], 'best', 'record']) + '.txt')

	print('Best-generalizing model is "' + textcolors.OKBLUE + best_model_test + textcolors.ENDC + '".')
	if params['report-everything']:									#  A confusion matrix and test-set performance already exist. Clone them.
		shutil.copy(best_model_test[:-4]+'-confusionmatrix.txt', \
		            '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], 'bestGeneralizing', 'confusionmatrix']) + '.txt')
		shutil.copy(best_model_test[:-4]+'-record.txt', \
		            '-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str'], 'bestGeneralizing', 'record']) + '.txt')

	#model.load_state_dict(torch.load(best_model, weights_only=False))

	#print('Computing test-set confusion matrix.')

	#test_and_report(test_dataloader, model, loss_fn, device, params, best_model, test_data)

	print('\n')

	if params['plot-loss']:
																	#  Collect all bookmarks.
		bookmarks = [x for x in os.listdir() if x.startswith('-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str']])) \
		                                    and x.endswith('.txt') and '-record.txt' not in x and '-confusionmatrix.txt' not in x]
																	#  Sort BOOKMARKS ascending according to epoch.
		bookmarks = sorted(bookmarks, key=lambda x: int(x.split('-')[-1].split('.')[0]))

		records = [x for x in os.listdir() if x.startswith('-'.join(['GraspSuccess', params['principal-estimator'], params['objects-set-name'], params['gripper-str'], params['mode-str']])) \
		                                  and 'best' not in x and x.endswith('-record.txt')]
																	#  Sort RECORDS ascending according to epoch.
		records = sorted(records, key=lambda x: int(x.split('-')[-2].split('.')[0]))

		train_loss = []
		test_loss = []

		test_acc = []

		epochs = [x for x in range(0, len(bookmarks))]
		for i in epochs:
			fh = open(bookmarks[i], 'r')
			lines = [x for x in fh.readlines() if x[0] != '#']
			fh.close()

			train_loss.append( float(lines[0].strip().split(':')[1].strip()) )
			test_loss.append( float(lines[1].strip().split(':')[1].strip()) )

			M = np.zeros((2, 2), dtype=np.uint16)
			fh = open(records[i], 'r')								#  Raw prediction <t> Rounded prediction <t> Ground-truth label ...
			lines = [x for x in fh.readlines() if x[0] != '#']
			for line in lines:
				arr = line.strip().split('\t')
				pred = int(arr[1])
				gt = int(arr[2])
				M[gt, pred] += 1
			test_acc.append( float(M.trace()) / float(M.sum()) )	#  Push accuracy
			fh.close()

		plt.plot(epochs, train_loss, color='#00843D')				#  Training loss in green
		plt.plot(epochs, test_loss, color='#003DA5')				#  Validation loss in blue

		plt.title('-'.join(best_model[:-4].split('-')[:-1]) + ' Training')
		figure_name = '-'.join(best_model[:-4].split('-')[:-1]) + '-loss.png'
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.savefig(figure_name)
		plt.clf()

		plt.plot(epochs, train_loss, color='#00843D')				#  Training loss (only) in green

		plt.title('-'.join(best_model[:-4].split('-')[:-1]) + ' Training')
		figure_name = '-'.join(best_model[:-4].split('-')[:-1]) + '-trainingLoss.png'
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.savefig(figure_name)
		plt.clf()

		plt.plot(epochs, test_acc, color='#00ff00')					#  Accuracy in bright green
		plt.title('-'.join(best_model[:-4].split('-')[:-1]) + ' Accuracy')
		figure_name = '-'.join(best_model[:-4].split('-')[:-1]) + '-accuracy.png'
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.savefig(figure_name)
		plt.clf()

		plt.plot(epochs, learning_rate, color='#DA291C')			#  Learning rate in red
		plt.title('-'.join(best_model[:-4].split('-')[:-1]) + ' Learning Rate')
		figure_name = '-'.join(best_model[:-4].split('-')[:-1]) + '-lr.png'
		plt.xlabel("Epochs")
		plt.ylabel("Learning Rate")
		plt.savefig(figure_name)
		plt.clf()

	return

def train(dataloader, model, loss_fn, optimizer, scheduler, device, params):
	ctr = 0
	prev_ctr = 0
	num_samples = len(dataloader.dataset)
	num_batches = len(dataloader)
	format_string = '{:.'+str(params['prog-bar-loss-prec'])+'f}'

	model.train()
	avg_batch_time = 0.0											#  Running average time per batch in seconds.
	curr_mean_training_loss = 0.0

	for batch, (X, y) in enumerate(dataloader):
		optimizer.zero_grad()										#  Reset.

		batch_time_start = time.process_time()						#  Start time for one batch: seconds.
		X, y = X.to(device), y.to(device)

		pred = model(X)												#  Compute prediction error.
		loss = loss_fn(pred, y.unsqueeze(1))						#  Compute loss.
		curr_mean_training_loss += loss.item()

		loss.backward()												#  Backpropagate.
		optimizer.step()											#  Tune.
		batch_time_end = time.process_time()						#  Stop time for one batch: seconds.

		avg_batch_time = (ctr * avg_batch_time + (batch_time_end - batch_time_start)) / (ctr + 1)
		eta = avg_batch_time * (num_batches - ctr)					#  E.T.A. in seconds.

		if eta > 3600:												#  Display hours.
			prog_output_str = ' ' + format_string.format(curr_mean_training_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta/3600) + 'h]'
		elif eta > 60:												#  Display minutes.
			prog_output_str = ' ' + format_string.format(curr_mean_training_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta/60) + 'm]'
		else:														#  Display seconds.
			prog_output_str = ' ' + format_string.format(curr_mean_training_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta) + 's]'

		prev_ctr = int(np.ceil(float(ctr * params['batch-size']) / float(num_samples) * float(params['prog-bar-len'] - len(prog_output_str)))) - 2
		sys.stdout.write('\r[' + textcolors.OKGREEN + '='*prev_ctr + prog_output_str + textcolors.ENDC)
		sys.stdout.flush()
		ctr += 1

	scheduler.step()												#  Advance the learning-rate scheduler.

	prog_output_str = ' 100%]'
	sys.stdout.write('\r[' + textcolors.OKGREEN + '='*(params['prog-bar-len'] - len(prog_output_str) - 2) + textcolors.ENDC + prog_output_str)
	sys.stdout.flush()
	print('')														#  Clear the line.
	return curr_mean_training_loss / num_batches

def test(dataloader, model, loss_fn, device, params):
	ctr = 0
	prev_ctr = 0
	num_samples = len(dataloader.dataset)
	num_batches = len(dataloader)
	format_string = '{:.'+str(params['prog-bar-loss-prec'])+'f}'

	model.eval()
	avg_batch_time = 0.0											#  Running average time per batch in seconds.
	curr_mean_test_loss = 0.0

	with torch.no_grad():
		for X, y in dataloader:
			batch_time_start = time.process_time()					#  Start time for one batch: seconds.
			X, y = X.to(device), y.to(device)
			pred = model(X)
			loss = loss_fn(pred, y.unsqueeze(1))					#  Compute loss.
			curr_mean_test_loss += loss.item()

			batch_time_end = time.process_time()					#  Stop time for one batch: seconds.

			avg_batch_time = (ctr * avg_batch_time + (batch_time_end - batch_time_start)) / (ctr + 1)
			eta = avg_batch_time * (num_batches - ctr)				#  E.T.A. in seconds.

			if eta > 3600:											#  Display hours.
				prog_output_str = ' ' + format_string.format(curr_mean_test_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta/3600) + 'h]'
			elif eta > 60:											#  Display minutes.
				prog_output_str = ' ' + format_string.format(curr_mean_test_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta/60) + 'm]'
			else:													#  Display seconds.
				prog_output_str = ' ' + format_string.format(curr_mean_test_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta) + 's]'

			prev_ctr = int(np.ceil(float(ctr * params['batch-size']) / float(num_samples) * float(params['prog-bar-len'] - len(prog_output_str)))) - 2
			sys.stdout.write('\r[' + textcolors.OKBLUE + '='*prev_ctr + prog_output_str + textcolors.ENDC)
			sys.stdout.flush()
			ctr += 1

	prog_output_str = ' 100%]'
	sys.stdout.write('\r[' + textcolors.OKBLUE + '='*(params['prog-bar-len'] - len(prog_output_str) - 2) + textcolors.ENDC + prog_output_str)
	sys.stdout.flush()
	print('')														#  Clear the line.

	return curr_mean_test_loss / num_batches

def test_and_report(dataloader, model, loss_fn, device, params, network_name, dataset):
	ctr = 0
	prev_ctr = 0
	num_samples = len(dataloader.dataset)
	num_batches = len(dataloader)
	format_string = '{:.'+str(params['prog-bar-loss-prec'])+'f}'

	timestamp_str = time.strftime('%l:%M%p %Z on %b %d, %Y')
																	#  GraspSuccess-<principal>-<object>-<gripper>-<mode>-<epoch>-record.txt
	fh = open(network_name[:-4]+'-record.txt', 'w')
	fh_confmat = open(network_name[:-4]+'-confusionmatrix.txt', 'w')

	fh.write('#  Grasp Success Predictor Network "'+network_name+'"\'s predictions on test set.\n')
	fh.write('#  PRINCIPAL:   '+params['principal-estimator']+'\n')
	fh.write('#  OBJECT(S):  {'+','.join(params['object-strs'])+'}\n')
	fh.write('#  OBJ.SET:     '+params['objects-set-name']+'\n')
	fh.write('#  GRIPPER(S): {'+params['gripper-str']+'}\n')
	fh.write('#  MODE:        '+params['mode-str']+'\n')
	fh.write('#  Created ' + timestamp_str + '.\n')
	fh.write('#    python3 ' + ' '.join(sys.argv) + '\n')
	fh.write('#  LINE FORMAT:' + '\n')
	fh.write('#    Raw prediction <t> Rounded prediction <t> Ground-truth label. <t> Dataset <t> Scene <t> Frame <t> Visibility <t> Gripper\n')

	fh_confmat.write('#  Confusion matrix for Grasp Success Predictor Network "'+network_name+'".\n')
	fh_confmat.write('#  PRINCIPAL:   '+params['principal-estimator']+'\n')
	fh_confmat.write('#  OBJECT(S):  {'+','.join(params['object-strs'])+'}\n')
	fh_confmat.write('#  OBJ.SET      '+params['objects-set-name']+'\n')
	fh_confmat.write('#  GRIPPER(S): {'+params['gripper-str']+'}\n')
	fh_confmat.write('#  MODE:        '+params['mode-str']+'\n')

	conf_mat = np.zeros((2, 2), dtype=np.uint16)

	model.eval()
	avg_batch_time = 0.0											#  Running average time per batch in seconds.
	curr_mean_test_loss = 0.0

	dataset_ctr = 0													#  This counter must be independent of the index over batches.

	with torch.no_grad():
		for X, y in dataloader:
			batch_time_start = time.process_time()					#  Start time for one batch: seconds.
			X, y = X.to(device), y.to(device)
			pred = model(X)
			loss = loss_fn(pred, y.unsqueeze(1))					#  Compute loss.
			curr_mean_test_loss += loss.item()

			batch_time_end = time.process_time()					#  Stop time for one batch: seconds.

			pred_cpu = [x[0] for x in pred.to('cpu').tolist()]		#  Back to host: reshape into list of length batch-size.
			y_cpu = y.to('cpu').tolist()							#  Back to host: reshape into list of length batch-size.

			for i in range(0, len(pred_cpu)):						#  For each sample in batch...
				pred_soft = pred_cpu[i]
				pred_hard = int(max(0.0, min(1.0, round(pred_soft))))
				gt = int(y_cpu[i])

				conf_mat[gt, pred_hard] += 1
																	#  Write to record.
				fh.write(str(pred_soft) + '\t' + str(pred_hard) + '\t' + str(int(gt)) + '\t' + '\t'.join( dataset.src[dataset_ctr] ) + '\n')
				dataset_ctr += 1

			avg_batch_time = (ctr * avg_batch_time + (batch_time_end - batch_time_start)) / (ctr + 1)
			eta = avg_batch_time * (num_batches - ctr)				#  E.T.A. in seconds.

			if eta > 3600:											#  Display hours.
				prog_output_str = ' ' + format_string.format(curr_mean_test_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta/3600) + 'h]'
			elif eta > 60:											#  Display minutes.
				prog_output_str = ' ' + format_string.format(curr_mean_test_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta/60) + 'm]'
			else:													#  Display seconds.
				prog_output_str = ' ' + format_string.format(curr_mean_test_loss / (ctr + 1)) + ', ETA ' + '{:.2f}'.format(eta) + 's]'

			prev_ctr = int(np.ceil(float(ctr * params['batch-size']) / float(num_samples) * float(params['prog-bar-len'] - len(prog_output_str)))) - 2
			sys.stdout.write('\r[' + textcolors.OKBLUE + '='*prev_ctr + prog_output_str + textcolors.ENDC)
			sys.stdout.flush()
			ctr += 1

	prog_output_str = ' 100%]'
	sys.stdout.write('\r[' + textcolors.OKBLUE + '='*(params['prog-bar-len'] - len(prog_output_str) - 2) + textcolors.ENDC + prog_output_str)
	sys.stdout.flush()
	print('')														#  Clear the line.

	fh_confmat.write('#  Created ' + timestamp_str + '.\n')
	fh_confmat.write('#    python3 ' + ' '.join(sys.argv) + '\n')
	fh_confmat.write('#  ACCURACY: ' + str(conf_mat.trace() / conf_mat.sum()) + '\n')
	fh_confmat.write('#  Rows: ground truth; Columns: predictions.\n')
	fh_confmat.write('#  \t0\t1\n')
	fh_confmat.write('0\t'+str(conf_mat[0, 0])+'\t'+str(conf_mat[0, 1])+'\n')
	fh_confmat.write('1\t'+str(conf_mat[1, 0])+'\t'+str(conf_mat[1, 1])+'\n')

	fh.close()
	fh_confmat.close()

	return curr_mean_test_loss / num_batches

def get_command_line_params():
	params = {}

	params['principal-estimator'] = None

	params['objects'] = []
	params['object-strs'] = []
	params['bop-objects'] = ['LMO1', 'LMO5', 'LMO6', 'LMO8', 'LMO9', 'LMO10', 'LMO11', 'LMO12', 'YCBV2', 'YCBV3', 'YCBV4', 'YCBV5', 'YCBV8', 'YCBV9', 'YCBV12']
	params['objects-set-name'] = None								#  Be able to refer to the set of objects you use. e.g. "all", "prismatics", etc.

	params['gripper'] = None
	params['gripper-str'] = None

	params['mode'] = PoseDifferenceMode.TRANS_ERR_3VEC_ROT_ERR_3VEC
	params['mode-str'] = 't3r3'
	params['mode-args'] = ['t1r1', 't3r3', 't1r3', 't3r1', 't4r4', 'add', 'wadd']

	params['epochs'] = 100

	params['optimizer'] = {}
	params['optimizer']['type'] = 'Adam'
																	#  With step-size 30 and gamma 0.1:
	params['learning-rate-schedule'] = {}
	params['learning-rate-schedule']['type'] = 'cosine'				#  Default to CosineAnnealingLR, 0.001 to 0.000001
	params['learning-rate-schedule']['initial'] = 0.001
	params['learning-rate-schedule']['min'] = 0.000001

	params['batch-size'] = 16
	params['resume'] = False										#  Whether to search for the latest checkpoint.
	params['plot-loss'] = False

	params['save-policy'] = 'improve'								#  In {'On Improvement', 'All', 'None'}
	params['save-policy-args'] = ['improve', 'all', 'none']

	params['prog-bar-len'] = os.get_terminal_size().columns			#  Total terminal width.
	params['prog-bar-loss-prec'] = 5								#  Numerical precision of progress bar's loss display.
	params['initial-epoch'] = 0
	params['report-everything'] = False

	params['prefer-gpu'] = 1										#  31aug24: sharing Viper with Liyan. She'll take GPU[0]; I'll take GPU[1].

	params['verbose'] = False
	params['helpme'] = False

	argtarget = None												#  Current argument to be set
																	#  Permissible setting flags
	flags = ['-principal', '-obj', '-objname', '-gripper', '-mode', '-epoch', '-epochs', '-optim', \
	         '-lrsched', '-lr', '-lrmin', '-lrrestart', '-lrfactor', '-lrstep', '-lrgamma', \
	         '-batch', '-resume', '-save', '-plot', '-report', \
	         '-v', '-?', '-help', '--help']
	for i in range(1, len(sys.argv)):
		if sys.argv[i] in flags:
			if sys.argv[i] == '-resume':
				params['resume'] = True
			elif sys.argv[i] == '-plot':
				params['plot-loss'] = True
			elif sys.argv[i] == '-report':
				params['report-everything'] = True
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
				elif argtarget == '-obj':
					if argval.upper().replace('-', '') == 'LMO1':
						params['objects'].append( BOPObject.LMO1 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO5':
						params['objects'].append( BOPObject.LMO5 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO6':
						params['objects'].append( BOPObject.LMO6 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO8':
						params['objects'].append( BOPObject.LMO8 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO9':
						params['objects'].append( BOPObject.LMO9 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO10':
						params['objects'].append( BOPObject.LMO10 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO11':
						params['objects'].append( BOPObject.LMO11 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'LMO12':
						params['objects'].append( BOPObject.LMO12 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV2':
						params['objects'].append( BOPObject.YCBV2 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV3':
						params['objects'].append( BOPObject.YCBV3 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV4':
						params['objects'].append( BOPObject.YCBV4 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV5':
						params['objects'].append( BOPObject.YCBV5 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV8':
						params['objects'].append( BOPObject.YCBV8 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV9':
						params['objects'].append( BOPObject.YCBV9 )
						params['object-strs'].append( argval.upper().replace('-', '') )
					elif argval.upper() == 'YCBV12':
						params['objects'].append( BOPObject.YCBV12 )
						params['object-strs'].append( argval.upper().replace('-', '') )
				elif argtarget == '-objname':
					params['objects-set-name'] = argval
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
					elif argval.lower() == 't4r4':
						params['mode'] = PoseDifferenceMode.TRANS_3VEC_ROT_3VEC_TRANS_SCALAR_ROT_SCALAR
						params['mode-str'] = argval.lower()
					elif argval.lower() == 'add':
						params['mode'] = PoseDifferenceMode.AVERAGE_DISTANCE_DISTINGUISHABLE
						params['mode-str'] = argval.lower()
					elif argval.lower() == 'wadd':
						params['mode'] = PoseDifferenceMode.WEIGHTED_AVERAGE_DISTANCE_DISTINGUISHABLE
						params['mode-str'] = argval.lower()
				elif argtarget == '-epoch' or argtarget == '-epochs':
					params['epochs'] = max(int(argval), 1)

				elif argtarget == '-optim':
					if argval.lower() == 'sgd':
						params['optimizer']['type'] = 'SGD'
					elif argval.lower()[:4] == 'adam':
						params['optimizer']['type'] = 'Adam'
					elif argval.lower()[:4] == 'adag':
						params['optimizer']['type'] = 'Adagrad'

				elif argtarget == '-lrsched':
					if argval.lower()[:4] == 'cosw':
						params['learning-rate-schedule']['type'] = 'cosine-warm-restarts'
						params['learning-rate-schedule']['initial'] = 0.001
						params['learning-rate-schedule']['iter-to-restart'] = 30
						params['learning-rate-schedule']['factor'] = 1
						params['learning-rate-schedule']['min'] = 0.000001
					elif argval.lower()[:3] == 'cos':
						params['learning-rate-schedule']['type'] = 'cosine'
						params['learning-rate-schedule']['initial'] = 0.001
						params['learning-rate-schedule']['min'] = 0.000001
					elif argval.lower()[:4] == 'step':
						params['learning-rate-schedule']['type'] = 'step'
						params['learning-rate-schedule']['initial'] = 0.001
						params['learning-rate-schedule']['gamma'] = 0.1
						params['learning-rate-schedule']['step'] = 30
				elif argtarget == '-lr':
					params['learning-rate-schedule']['initial'] = float(argval)
				elif argtarget == '-lrmin' and (params['learning-rate-schedule']['type'] == 'cosine' or params['learning-rate-schedule']['type'] == 'cosine-warm-restarts'):
					params['learning-rate-schedule']['min'] = float(argval)
				elif argtarget == '-lrrestart' and params['learning-rate-schedule']['type'] == 'cosine-warm-restarts':
					params['learning-rate-schedule']['iter-to-restart'] = int(argval)
				elif argtarget == '-lrfactor' and params['learning-rate-schedule']['type'] == 'cosine-warm-restarts':
					params['learning-rate-schedule']['factor'] = int(argval)
				elif argtarget == '-lrstep' and params['learning-rate-schedule']['type'] == 'step':
					params['learning-rate-schedule']['step'] = int(argval)
				elif argtarget == '-lrgamma' and params['learning-rate-schedule']['type'] == 'step':
					params['learning-rate-schedule']['gamma'] = float(argval)

				elif argtarget == '-batch':
					params['batch-size'] = max(int(argval), 1)
				elif argtarget == '-save' and argval in params['save-policy-args']:
					params['save-policy'] = argval

	return params

def usage():
	print('Train for a single (principal, {objects}, gripper); test on that same (principal, {objects}, gripper):')
	print('================================================================================================')
	print('python3 train_NObj1Grip.py -principal EPOS -objname all -obj LMO1 -obj LMO5 -obj LMO6 -obj LMO8 -obj LMO9 -obj LMO12 -obj YCBV2 -obj YCBV3 -obj YCBV4 -obj YCBV5 -obj YCBV8 -obj YCBV9 -obj YCBV12 -gripper par -epochs 400 -lr 0.0001 -batch 16 -mode t3r3 -v -plot -report -save all')
	print('python3 train_NObj1Grip.py -principal GDRNPP -objname all -obj LMO1 -obj LMO5 -obj LMO6 -obj LMO8 -obj LMO9 -obj LMO12 -obj YCBV2 -obj YCBV3 -obj YCBV4 -obj YCBV5 -obj YCBV8 -obj YCBV9 -obj YCBV12 -gripper par -epochs 400 -lr 0.0001 -batch 16 -mode t3r3 -v -plot -report -save all')
	print('python3 train_NObj1Grip.py -principal ZebraPose -objname all -obj LMO1 -obj LMO5 -obj LMO6 -obj LMO8 -obj LMO9 -obj LMO12 -obj YCBV2 -obj YCBV3 -obj YCBV4 -obj YCBV5 -obj YCBV8 -obj YCBV9 -obj YCBV12 -gripper par -epochs 400 -lr 0.0001 -batch 16 -mode t3r3 -v -plot -report -save all')
	return

if __name__ == '__main__':
	main()