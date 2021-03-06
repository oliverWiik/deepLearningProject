from genericpath import exists
import torch
import numpy as np
import torch.nn as nn
import os
from typing import List
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AdamW
from transformers import Adafactor
from datetime import datetime
import pickle
#from IPython.display import clear_output


from ClassesAndFunctions import *

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

##### Setup #####

batch_size = 5
num_epoch = 1

model = T5ForConditionalGeneration.from_pretrained("ufal/byt5-small-multilexnorm2021-da")
#model.load_state_dict(torch.load("model_106866")) # Only for a specific continuation of a stopped session

optimizer = AdamW(model.parameters(), lr=5e-5)
print("Optimizer: Adam")
#optimizer = Adafactor(model.parameters())
#print("Optimizer: Adafactor")
criterion = nn.CrossEntropyLoss()


reloadData = True
if reloadData:
	dataset = MultiLexDataset(path_to_files=["final_nst.txt", "final_audiobooks.txt"], only_include_corrections=False, short_data=False)
	with open('dataset.pickle', 'wb') as f:
		pickle.dump(dataset, f)
else:
	with open('dataset.pickle', 'rb') as f:
		dataset = pickle.load(f)

# Use with a datalodaer
tokenizer = AutoTokenizer.from_pretrained("ufal/byt5-small-multilexnorm2021-da")
trainloader = DataLoader(dataset.train, batch_size=batch_size, collate_fn=CollateFunctor(tokenizer))
validationloader = DataLoader(dataset.validation, batch_size=1, collate_fn=CollateFunctor(tokenizer))
testloader = DataLoader(dataset.test, batch_size=1, collate_fn=CollateFunctor(tokenizer))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


num_training_steps = int(num_epoch * np.ceil(len(dataset.train)/batch_size))


##### Freeze model #####

for name, param in model.named_parameters():
  	param.requires_grad = True



##### Training #####

print("Initiate training")
progress_bar = tqdm(range(num_training_steps), position=0)
model.train()

eval_every = round(num_training_steps*0.01) # 1%
first = True
steps_training_plot = []
steps_validation_plot = []
trainingLossArr = [] 
validationLossArr = []
current_training_batch = 0

with open("lossData.txt", "a") as file_object:
				file_object.write('Samples in per epoch:\t' + str(np.ceil(len(dataset.train)/batch_size)) + '\t Nb epochs:\t' + str(num_epoch) + '\n')

for epoch in range(num_epoch):
	for batch_idx, batch in enumerate(trainloader):
		#if batch_idx < 106866:  # Only for a specific continuation of a stopped session
		#	continue
		
		
		current_training_batch += 1
		steps_training_plot.append(current_training_batch)
		model.train()
      
      	# Move batch to GPU
		batch = {k: v.to(device) for k, v in batch.items()}

      #Make prediction with the model
		output = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"],decoder_attention_mask=batch["decoder_attention_mask"])

      # Zero_grad, backwards and optimizer step
		optimizer.zero_grad()
		loss = output.loss
		trainingLossArr.append(loss.item())

		loss.backward()
		optimizer.step()


      # Do validation
		if batch_idx % eval_every == 0 or batch_idx == len(trainloader):
			progress_bar_eval = tqdm(total=len(dataset.validation), position=1)
			steps_validation_plot.append(current_training_batch)
			model.eval()
			validationLoss2Mean = 0
			for num, batch_validation in enumerate(validationloader):
				batch_validation = {k: v.to(device) for k, v in batch_validation.items()}
				with torch.no_grad():
					output = model(**batch_validation)

				validationLoss2Mean += output.loss.item()

				progress_bar_eval.update(1)

			validationLoss2Mean /= len(validationloader)

			modelName = "models/model_" + str(current_training_batch)
			
			if not os.path.exists("models"):
				os.mkdir("models")
			
			if batch_idx >= 7*10**4 and batch_idx % eval_every*3 == 0:
				torch.save(model.state_dict(),modelName)

			EvaluatedMetricsTest = testsetAgainstNLPMetrics(dataset, tokenizer, model, device)

			with open("lossData.txt", "a") as file_object:
				file_object.write(str(datetime.now()) + '\t' + str(current_training_batch) + '\t' + str(np.mean(trainingLossArr)) + '\t' + 
				str(validationLoss2Mean) + '\t' + str(EvaluatedMetricsTest.errorCalc.errorMeanMetrics["wer"]) + '\t' + 
				str(EvaluatedMetricsTest.errorCalc.errorMeanMetrics["bleu"]) + '\t' + str(EvaluatedMetricsTest.errorCalc.errorMeanMetrics["gleu"]) + '\n')
			
			trainingLossArr = []

		
		progress_bar.update(1)
  
	# kill batch
	del batch




















