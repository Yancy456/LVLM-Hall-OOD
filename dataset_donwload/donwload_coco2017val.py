from datasets import load_dataset
import os
from tqdm import tqdm
# Load the COCO 2017 validation dataset
dataset = load_dataset("nielsr/coco-panoptic-val2017")


# Create a directory to save the images
output_dir = '/root/autodl-tmp/coco_val_images'
os.makedirs(output_dir, exist_ok=True)

# Iterate through the dataset and save the images
# Replace 'train' with the correct split if needed
for i, example in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
    path = os.path.join(output_dir, f'{i}.jpg')
    # example['image'].save(path)

print("All images have been saved!")
