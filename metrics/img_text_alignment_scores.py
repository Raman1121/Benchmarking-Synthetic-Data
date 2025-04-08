from health_multimodal.text import get_bert_inference
from health_multimodal.image import get_image_inference
"""
We calculate the image-text aignment score using the health_multimodal toolbox (https://github.com/microsoft/hi-ml)
"""

from health_multimodal.vlp import ImageTextInferenceEngine
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Image-Text Alignment Score b/w Real Prompt and Synthetic Images")
    parser.add_argument("--results_csv", type=str, default=None, , help="CSV File containing the path to synthetic images and corresponding prompts.")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory containing the synthetic images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("prompts_col", type=str, default="prompt", help="Column name in the CSV file containing the prompts.")
    parser.add_argument("img_col", type=str, default="img_savename", help="Column name in the CSV file containing the image paths.")
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# A class DummyDataset that accepts a CSV file containing the path to synthetic images and corresponding prompts and returns the image and text data.
class DummyDataset(Dataset):
    def __init__(self, args, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = Path(self.img_dir) / self.data.iloc[idx][self.args.img_col]
        text = self.data.iloc[idx][self.args.prompts_col]
        image = Image.open(img_path).convert("RGB")
        return image, text

def main(args):
    text_inference = get_bert_inference()
    image_inference = get_image_inference()
    image_text_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )

    dataset = DummyDataset(args, args.results_csv, args.img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    all_scores = []

    for images, texts in tqdm(dataloader):
        images = images.to("cuda")
        texts = texts.to("cuda")

        with torch.no_grad():
            scores = image_text_inference(images, texts)
            all_scores.append(scores.cpu().numpy())
    all_scores = np.concatenate(all_scores, axis=0)
    all_scores = pd.DataFrame(all_scores, columns=["image_text_score"])
    all_scores.to_csv("img_text_scores.csv", index=False)
    print("Image-Text Alignment Scores saved to img_text_scores.csv")
    print("Image-Text Alignment Scores: ", all_scores)