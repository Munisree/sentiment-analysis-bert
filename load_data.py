from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Print a few examples
print(dataset['train'][0])
