import torch
import numpy as np
import torch.nn as nn

from typing import List
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AdamW
#from IPython.display import clear_output


from ClassesAndFunctions import *

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

##### Setup #####

batch_size = 1
num_epoch = 1

model = T5ForConditionalGeneration.from_pretrained("ufal/byt5-small-multilexnorm2021-da")
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


dataset = MultiPlexDataset(path_to_files=["final_nst.txt", "final_audiobooks.txt"], only_include_corrections=False, short_data=True)


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
progress_bar = tqdm(range(num_training_steps))
model.train()

eval_every = 100
first = True
steps_training_plot = []
steps_validation_plot = []
trainingLossArr = [] 
validationLossArr = []
current_training_batch = 0

for epoch in range(num_epoch):
	for batch_idx, batch in enumerate(trainloader):
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
		if batch_idx % eval_every == 0 or batch_idx == num_training_steps:
			steps_validation_plot.append(current_training_batch)
			model.eval()
			validationLoss2Mean = 0
			for num, batch_validation in enumerate(validationloader):
				batch_validation = {k: v.to(device) for k, v in batch_validation.items()}
				with torch.no_grad():
					output = model(**batch_validation)

				validationLoss2Mean += output.loss.item()

			validationLoss2Mean /= len(validationloader)

			with open("lossData.txt", "a") as file_object:
				file_object.write(str(current_training_batch) + '\t' + str(np.mean(trainingLossArr)) + '\t' + str(validationLoss2Mean) + '\n')
			trainingLossArr = []
		
		progress_bar.update(1)
  
	# kill batch
	del batch


EvaluatedMetricsTest = testsetAgainstMLPMetrics(dataset)
print(EvaluatedMetricsTest.errorMeanMetrics)


## Write metrics in a txt file

with open('metrics.txt', 'w') as f:
    f.write(str(EvaluatedMetricsTest.errorMeanMetrics))




















