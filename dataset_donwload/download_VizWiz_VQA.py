from datasets import load_dataset
from tqdm import tqdm
import os
# Load the COCO 2017 validation dataset
dataset = load_dataset("/root/autodl-fs/VizWiz")


# Create a directory to save the images
output_dir = '/root/autodl-fs/VizWiz'
os.makedirs(output_dir, exist_ok=True)


for i, example in tqdm(enumerate(dataset['test']), total=len(dataset['test'])):
    path = os.path.join(output_dir, f'{i}.jpg')
    # example['image'].save(path)
