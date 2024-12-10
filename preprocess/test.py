from datasets import load_dataset
import os

# Load a dataset from the Hugging Face Hub
dataset = load_dataset(
    "/root/autodl-fs/coco2014", num_proc=8)

# Print the first few examples
print(dataset[:5])
