import torch
use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

## Upload a sample txt to confirm upload 

with open('results.txt', 'w') as f:
	f.write("This is a test run from a HPC node..")

import  git 

repo = git.Repo('deepLearningProject')
repo.git.add('results.txt')
repo.git.commit('-m', 'commit message from python script', author='mortenvorborg@hotmail.com')
origin = repo.remote(name='origin')
origin.push()