use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")
