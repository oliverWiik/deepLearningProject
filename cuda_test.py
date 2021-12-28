import torch
use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

## Upload a sample txt to confirm upload 

with open('results.txt', 'w') as f:
	f.write("This is a test run from a HPC node..")


import git, os
repo = git.Repo.clone_from("https://github.com/oliverWiik/deepLearningProject.git", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'repo'), branch='master')
#repo.index.add(["results.txt"])
#repo.index.commit("This is a test commit message form HPC")
master = repo.heads.master
master.add(["results.txt"])
master.commit("Tester test")
master.push()

#from git import Repo
#Repo.clone_from("https://github.com/oliverWiik/deepLearningProject.git", "Desktop")

#import git 

#repo = git.Repo('deepLearningProject')
#git.add('results.txt')
#git.commit('-m', 'commit message from python script', author='mortenvorborg@hotmail.com')

#origin = remote(name='origin')
#origin.push()