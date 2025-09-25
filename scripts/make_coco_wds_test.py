import torch
from tqdm import tqdm
import webdataset as wds
from torchvision import transforms
import os
import glob

# This guard is crucial for multiprocessing (num_workers > 0) to work correctly.
if __name__ == "__main__":
    # --- 1. Configuration ---
    # Adjust this path to point to your training shards
    data_dir = "data/coco_wds/train"
    # The pattern matches all .tar files in the directory
    urls = os.path.join(data_dir, "coco-train-*.tar")
    urls = sorted(glob.glob(urls))
    print(f"Found {len(urls)} shards.")
    
    batch_size = 16
    num_workers = 4 # Set to 0 if you want to debug on the main process

    # --- 2. Define Transformations ---
    # Define the transformations for the image.
    # Normalization to [-1, 1] is common for GANs and VAEs.
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])

    # For segmentation masks, we only want to resize.
    # Crucially, use NEAREST interpolation to avoid creating new class labels at boundaries.
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
    ])

    # --- 3. Build the WebDataset Pipeline ---
    # For true shuffling, it's best to shuffle the order of the .tar files (shards)
    # in addition to shuffling the samples within them.
    dataset = wds.WebDataset(urls, resampled=True)
    
    # a. Shuffle the shards. The buffer determines how many shards are considered at once.
    dataset = dataset.shuffle(100)
    
    # b. Decode the raw data from the .tar file.
    #    'pil' will decode .jpg/.png to PIL Images and .json to Python objects.
    dataset = dataset.decode("pil")

    # c. Shuffle the samples within each shard. A larger buffer gives better shuffling.
    dataset = dataset.shuffle(1000)
    
    # d. Select the data streams and apply transformations.
    #    This creates a tuple (image, mask, caption) for each sample.
    dataset = dataset.to_tuple("jpg", "png", "txt")
    dataset = dataset.map_tuple(image_transform, mask_transform, lambda x: x) # Pass captions through

    # --- 4. Create the DataLoader ---
    # WebDataset is already iterable. We wrap it in a DataLoader to handle
    # batching, parallel loading, and other utilities.
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True # Good practice for GPU training
    )
    
    # To make it behave like a standard PyTorch loader, we set its length.
    # This is useful for progress bars like tqdm.
    # You need to know the total number of samples in your dataset.
    # Let's assume the COCO train set has 118,287 images.
    loader = loader.with_length(118287 // batch_size)

    # --- 5. Demonstrate Usage ---
    print(f"DataLoader created. Iterating through a few batches with {num_workers} workers...")
    
    for i, (images, masks, captions) in tqdm(enumerate(loader)):
        ...
        
    print("\nDemonstration finished.")
