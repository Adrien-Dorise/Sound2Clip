'''
 Main file to launch and check all dragonflAI's examples 
 Author: Edouard Villain (evillain@lrtechnologies.fr) - LR Technologies
 Created: May 2024
 Last updated: Edouard Villain - May 2024
'''
import os 
import json 

def print_check(name, ret, expected):
	"""print_check : print if name get its expected results

	Args:
		name (str): name of checked criteria 
		ret (float): results 
		expected (list): [results_min, results_max]
	"""
	if ret >= expected[0] and ret <= expected[1]:
		print('\033[92m {} passed -> {} in {} \033[00m'.format(name, ret, expected))
	else:
		print('\033[91m {} failed -> {} not in {} \033[00m'.format(name, ret, expected))

def check_ret(ret, expected):
	"""check_ret: comparison between results and expected 

	Args:
		ret (dict): dict of criteria / value 
		expected (dict): dict of criteria / value 
	"""
	#print('\tChecking : {}'.format(expected['name']))
	if float(expected['acc_train']) != 0.0:
		train_acc_ret = float(ret['acc_train'])
		val_acc_ret   = float(ret['acc_val'])
		train_acc_expected = [float(expected['acc_train']) - float(expected['acc_tol']) , 100.0]
		val_acc_expected = [float(expected['acc_val']) - float(expected['acc_tol']) , 100.0]
		print_check('training accuracy', train_acc_ret, train_acc_expected)
		print_check('val accuracy', val_acc_ret, val_acc_expected)
		
	train_loss_ret = float(ret['loss_train'])
	val_loss_ret   = float(ret['loss_val'])
	train_loss_expected = [0.0, float(expected['loss_train']) + float(expected['loss_tol'])]
	val_loss_expected = [0.0, float(expected['loss_val']) + float(expected['loss_tol'])]
	print_check('training loss', train_loss_ret, train_loss_expected)
	print_check('val loss', val_loss_ret, val_loss_expected)

        
        
def get_results(name):
	"""get_results : get results 

	Args:
		name (str): filename of csv results file 

	Returns:
		dict: dict of criteria / value 
	"""
	with open(name, 'r') as openfile:
		# Reading from json file
		return json.load(openfile)
        

def get_expected_results(path, name):
	"""get_expected_results : get expected results 

	Args:
		path (str): path to file 
		name (str): name of the experiment 

	Returns:
		dict: dict of criteria / value for current experiment 
	"""
	
	with open(path, 'r') as openfile:
		# Reading from json file
		return json.load(openfile)[name]


if __name__ == "__main__":
    # if RETRAIN is True, all models will be retrain 
    # otherwise, it will use last results 
	RETRAIN = True 
	# get current directory
	pwd = os.getcwd()
	# log file 
	log_file = pwd + '/examples/neural_network/validation.log'
	# list files and directories inside examples 
	d = os.listdir('./examples/neural_network')
	# foreach name 
	for name in d:
		# if name is a directory 
		if os.path.isdir(pwd + '/examples/neural_network/' + name):
			# retrain model if needed 
			if RETRAIN:
				cmd = 'python3 ./examples/neural_network/{}/main.py >> {}'.format(name, log_file)
				print('Training examples : {}'.format(name))
				os.system(cmd)
			else:
				print('Skip training... Using last training\'s results...')
			# get results and expected results 
			expected = get_expected_results(pwd + '/examples/neural_network/expected_results.json', name) 
			ret = get_results(pwd + '/examples/neural_network/' + name + '/results/end_train_results.json') 
			# check and print 
			check_ret(ret, expected)