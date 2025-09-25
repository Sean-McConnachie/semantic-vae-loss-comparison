import os
import webdataset as wds
from main import instantiate_from_config, DataModuleFromConfig
from omegaconf import OmegaConf
from tqdm.notebook import tqdm
import json

OUTPUT_DIR = "data/cocostuffthings_wds"
SAMPLES_PER_SHARD = 1000
UNIQUE_KEY = "filename_"
DATA_CONFIG = """
target: main.DataModuleFromConfig
params:
    num_workers: 0
    batch_size: 16
    train:
      target: taming.data.coco.CocoImagesAndCaptionsTrain
      params:
        size: 296
        crop_size: 256
        onehot_segmentation: true
        use_stuffthing: true
    validation:
      target: taming.data.coco.CocoImagesAndCaptionsValidation
      params:
        size: 256
        crop_size: 256
        onehot_segmentation: true
        use_stuffthing: true
"""

data_cfg = OmegaConf.create(DATA_CONFIG)

data: DataModuleFromConfig = instantiate_from_config(data_cfg)
data.prepare_data()
data.setup()

loaders = [
    ("train", data.train_dataloader()), 
    ("validation", data.val_dataloader())
]

for phase, loader in loaders:
    shard_pattern = os.path.join(OUTPUT_DIR, phase, "coco-%06d.tar")
    os.makedirs(os.path.dirname(shard_pattern), exist_ok=True)
    with wds.ShardWriter(shard_pattern, maxcount=SAMPLES_PER_SHARD) as sink:
        for sample in tqdm(loader.dataset):
            key = os.path.splitext(os.path.basename(sample[UNIQUE_KEY]))[0]

            with open(sample['img_path'], "rb") as f:
                image_data = f.read()
            
            with open(sample['seg_path'], "rb") as f:
                segmentation_data = f.read()
            
            caption_data = json.dumps(sample['caption']).encode("utf-8")

            output_sample = {
                "__key__": key,
                "jpg": image_data,
                "png": segmentation_data,
                "caption": caption_data
            }
            sink.write(output_sample)