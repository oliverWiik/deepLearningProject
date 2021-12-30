import torch
use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

## Upload a sample txt to confirm upload 

with open('results.txt', 'w') as f:
	f.write("This is a test run from a HPC node..")


import git, os
new_repo = git.Repo.init('new_repo')
git.Repo.clone_from('https://github.com/oliverWiik/deepLearningProject.git', 'DeeplearningHPCTest')
repo = git.Repo('DeeplearningHPCTest')
print(repo.remotes.origin.pull())
currentdir = os.getcwd()
targetfile = os.path.join(currentdir,'results.txt')
repo.index.add(['results.txt'])
repo.index.commit('initial test commit from HPC node')
print(repo.remotes.origin.push())

#repo = git.Repo.clone_from("https://github.com/oliverWiik/deepLearningProject.git", 'repo', branch='master')
#repo.index.add(["results.txt"])
#repo.index.commit("This is a test commit message form HPC")
#master = repo.heads.master
#master.add(["results.txt"])
#master.commit("Tester test")
#master.push()

#from git import Repo
#Repo.clone_from("https://github.com/oliverWiik/deepLearningProject.git", "Desktop")

#import git 

#repo = git.Repo('deepLearningProject')
#git.add('results.txt')
#git.commit('-m', 'commit message from python script', author='mortenvorborg@hotmail.com')

#origin = remote(name='origin')
#origin.push()