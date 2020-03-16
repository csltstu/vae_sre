import argparse

def get_args():
	parser = argparse.ArgumentParser(description='PyTorch Flows for sre')

        parser.add_argument(
                '--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

	parser.add_argument(
		'--no_cuda',
		action='store_true',
		default=False,
		help='disables CUDA training')

	parser.add_argument(
		'--device',
		default='0',
		help='cuda visible devices (default: 0)')

	parser.add_argument(
		'--batch_size',
		type=int,
		default=10000,
		help='input batch size for training (default: 10000)')

	parser.add_argument(
		'--epochs',
		type=int,
		default=100,
		help='number of epochs to train (default: 100)')

	parser.add_argument(
		'--ckpt_dir',
		default='ckpt',
		help='dir to save check points')

	args = parser.parse_args()
	
	return args

