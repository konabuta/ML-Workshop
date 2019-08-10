import torch
from torchvision import transforms
import os, json
import numpy as np

import utils
from transformer_net import TransformerNet

from azureml.core.model import Model

import time
import re
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def init():
	global model
	#model_path = os.path.join('picasso.pth')
	model_path = Model.get_model_path('picasso.pth')

	model = TransformerNet()
	state_dict = torch.load(model_path)
	# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
	for k in list(state_dict.keys()):
		if re.search(r'in\d+\.running_(mean|var)$', k):
			del state_dict[k]
	model.load_state_dict(state_dict)
	model.eval()

def run(input_data):
	try:
		input_data = torch.tensor(json.loads(input_data)['content'])
		with torch.no_grad():
			start = time.time()
			output = model(input_data)
			end = time.time()
		img = output[0].clone().clamp(0, 255).numpy()
		img = img.transpose(1, 2, 0).astype("uint8")
		result = {'stylized': img.tolist(), 'time': [end - start]}

	except Exception as e:
		result = {"error": str(e)}

	return json.dumps(result)
