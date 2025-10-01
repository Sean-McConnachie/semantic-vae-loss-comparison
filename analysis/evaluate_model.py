import os
import argparse
import csv

import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm, notebook as tqdm_notebook

from main import instantiate_from_config, DataModuleFromConfig
from taming.models.vqgan import VQSegmentationModel


DATA_CONFIG = """
target: main.DataModuleFromConfig
params:
    num_workers: 0
    batch_size: 8
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

MODEL_CONFIG = """
embed_dim: 256
n_embed: 1024
image_key: "segmentation"
n_labels: 183
ddconfig:
  double_z: false
  z_channels: 256
  resolution: 256
  in_channels: 183
  out_ch: 183
  ch: 128
  ch_mult:
  - 1
  - 1
  - 2
  - 2
  - 4
  num_res_blocks: 2
  attn_resolutions:
  - 16
  dropout: 0.0

lossconfig:
  target: taming.modules.losses.DummyLoss
"""


def segmentation_to_image(s, model):
    s = model.to_rgb(s)
    s = ((s+1.0)*127.5).clip(0,255).to(torch.uint8)
    s = s.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
    s = Image.fromarray(s)
    return s

def reconstruct(model, s):
    segmentation_rec, _ = model(s)
    segmentation_rec = torch.argmax(segmentation_rec, dim=1, keepdim=True)
    segmentation_rec = F.one_hot(segmentation_rec, num_classes=s.shape[1])
    segmentation_rec = segmentation_rec.squeeze(1).permute(0,3,1,2).to(torch.float32)
    return segmentation_rec

def calculate_miou(
    pred_indicies: torch.Tensor,
    tgt_indicies: torch.Tensor,
    num_classes: int,
    average: str = 'macro'
) -> float:
    miou_metric = torchmetrics.JaccardIndex(
        task='multiclass',
        num_classes=num_classes,
        average=average
    )

    miou_metric = miou_metric.to(pred_indicies.device)
    miou = miou_metric(pred_indicies, tgt_indicies)
    return miou.item()

def calculate_accuracy(
    pred_indices: torch.Tensor,
    tgt_indicies: torch.Tensor,
    num_classes: int,
    average: str | None = None
) -> torch.Tensor:
    accuracy_metric = torchmetrics.Accuracy(
        task='multiclass',
        num_classes=num_classes,
        average=average
    )
    accuracy_metric = accuracy_metric.to(pred_indices.device)
    accuracy = accuracy_metric(pred_indices, tgt_indicies)
    return accuracy

def calculate_f1(
    pred_indices: torch.Tensor,
    tgt_indicies: torch.Tensor,
    num_classes: int,
    average: str = 'macro'
) -> float:
    f1_metric = torchmetrics.F1Score(
        task='multiclass',
        num_classes=num_classes,
        average=average
    )
    f1_metric = f1_metric.to(pred_indices.device)
    f1 = f1_metric(pred_indices, tgt_indicies)
    return f1.item()

def calculate_precision(
    pred_indices: torch.Tensor,
    tgt_indicies: torch.Tensor,
    num_classes: int,
    average: str = 'macro'
) -> float:
    precision_metric = torchmetrics.Precision(
        task='multiclass',
        num_classes=num_classes,
        average=average
    )
    precision_metric = precision_metric.to(pred_indices.device)
    precision = precision_metric(pred_indices, tgt_indicies)
    return precision.item()

def calculate_recall(
    pred_indices: torch.Tensor,
    tgt_indicies: torch.Tensor,
    num_classes: int,
    average: str = 'macro'
) -> float:
    recall_metric = torchmetrics.Recall(
        task='multiclass',
        num_classes=num_classes,
        average=average
    )
    recall_metric = recall_metric.to(pred_indices.device)
    recall = recall_metric(pred_indices, tgt_indicies)
    return recall.item()


def main(
    data_cfg,
    model_cfg,
    run_name: str,
    ckpt_path: str,
    results_file: str = "results.csv"
):
    # load data loader
    print("loading data...")
    data: DataModuleFromConfig = instantiate_from_config(data_cfg)
    data.prepare_data()
    data.setup()
    # train_ldr = data.train_dataloader()
    val_ldr = data.val_dataloader()

    # load labels
    print("loading labels...")
    with open("data/cocostuffthings/labels.txt", "r") as f:
        labels = [line.strip().split(": ")[1] for line in f.readlines()]

    # create model and load checkpoint
    print("loading model...")
    model = VQSegmentationModel(**model_cfg)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    torch.set_grad_enabled(False)

    # reconstruct all validation images
    print("reconstructing validation set...")
    xs = []
    xrecs = []
    for i, batch in enumerate(tqdm(val_ldr)):
        x = batch["segmentation"].permute(0,3,1,2).to(dtype=torch.float32, device=model.device)
        if x.shape[1] != 183 and x.shape[2] != 256 and x.shape[3] != 256:
            print("skipping batch with wrong number of channels:", x.shape)
            continue

        xrec = reconstruct(model, x)

        xlbl = torch.argmax(x, dim=1)
        xreclbl = torch.argmax(xrec, dim=1)

        assert xlbl.min() >= 0 and xlbl.max() < 183, f"labels out of range: {xlbl.min()} - {xlbl.max()} for {i}"
        assert xreclbl.min() >= 0 and xreclbl.max() < 183, f"reconstructed labels out of range: {xreclbl.min()} - {xreclbl.max()} for {i}"

        xs.append(xlbl.detach().cpu())
        xrecs.append(xreclbl.detach().cpu())

    xs = torch.cat(xs, dim=0)
    xrecs = torch.cat(xrecs, dim=0)

    # calculate metrics
    print("calculating metrics...")
    results = {}
    results["run_name"] = run_name
    results["ckpt_path"] = ckpt_path
    print("calculating mIoU...")
    results["mIoU_macro"] = calculate_miou(xrecs, xs, num_classes=183, average='macro')
    print("calculating accuracy...")
    results["accuracy_macro"] = calculate_accuracy(xrecs, xs, num_classes=183, average='macro').item()
    print("calculating f1...")
    results["f1_macro"] = calculate_f1(xrecs, xs, num_classes=183, average='macro')
    print("calculating precision...")
    results["precision_macro"] = calculate_precision(xrecs, xs, num_classes=183, average='macro')
    print("calculating recall...")
    results["recall_macro"] = calculate_recall(xrecs, xs, num_classes=183, average='macro')
    print("calculating per class accuracy...")
    per_class_accuracy = calculate_accuracy(xrecs, xs, num_classes=183, average=None)
    for i in range(len(labels)):
        results[f"acc_{labels[i]}"] = per_class_accuracy[i].item()

    # save results
    print("writing results...")
    write_header = not os.path.exists(results_file)
    with open(results_file, "a") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--results_file", type=str, default="analysis/results.csv")
    args = parser.parse_args()

    data_cfg = OmegaConf.create(DATA_CONFIG)
    model_cfg = OmegaConf.create(MODEL_CONFIG)

    main(
        data_cfg,
        model_cfg,
        run_name=args.run_name,
        ckpt_path=args.ckpt_path,
        results_file=args.results_file
    )