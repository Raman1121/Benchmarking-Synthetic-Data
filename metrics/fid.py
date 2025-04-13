import os
import torch
import pandas as pd
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
import argparse
import warnings
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# RadDino Model: https://huggingface.co/microsoft/rad-dino
def load_RadDino_encoder():
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo)

    processor = AutoImageProcessor.from_pretrained(repo)

    return model, processor


class RadDinoFeatureExtractor(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def encode_image(self, model, processor, image):
        inputs = processor(images=image, return_tensors="pt")

        with torch.inference_mode():
            inputs = inputs.to(model.device)
            outputs = model(**inputs)
        cls_embeddings = outputs.pooler_output
        return cls_embeddings

    def forward(self, images):
        return self.encode_image(self.model, self.processor, images)


class ImageDataset(Dataset):
    def __init__(self, img_paths=None, transform=None):

        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        assert img_paths is not None
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            blank = Image.new("RGB", (299, 299), (0, 0, 0))
            return self.transform(blank)


def main(args):

    if args.debug:
        warnings.filterwarnings("ignore")
        print("Debug mode is ON. Make sure this behavior is intended.")

    # Set random seed for reproducibility
    seed_everything(42)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Paths to your CSV files
    synthetic_csv = args.synthetic_csv  # Contains 'prompt' and 'img_savename'
    real_csv = args.real_csv

    # Load real images
    real_df = pd.read_csv(real_csv)

    if args.debug:
        n_samples = args.debug_samples
        real_df = real_df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Drop rows with duplicate prompts
    real_df = real_df.drop_duplicates(subset=["annotated_prompt"]).reset_index(
        drop=True
    )
    real_df[args.real_img_col] = real_df[args.real_img_col].apply(
        lambda x: os.path.join(args.real_img_dir, x)
    )
    real_image_paths = real_df[
        args.real_img_col
    ].tolist()  # Adjust column name if needed

    # Load synthetic images
    synthetic_df = pd.read_csv(synthetic_csv)
    if args.debug:
        synthetic_df = synthetic_df.sample(n=n_samples, random_state=42).reset_index(
            drop=True
        )

    synthetic_df[args.synthetic_img_col] = synthetic_df[args.synthetic_img_col].apply(
        lambda x: os.path.join(args.synthetic_img_dir, x)
    )
    synthetic_image_paths = synthetic_df[
        args.synthetic_img_col
    ].tolist()  # The image path col in the CSV is 'img_savename'

    # Define transform for loading images
    # Note: torchmetrics FID expects images in range [0, 1] without normalization
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),  # Scales to [0, 1]
        ]
    )

    # Create datasets
    real_dataset = ImageDataset(img_paths=real_image_paths, transform=transform)
    synthetic_dataset = ImageDataset(
        img_paths=synthetic_image_paths, transform=transform
    )

    # Create dataloaders
    real_dataloader = DataLoader(
        real_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    synthetic_dataloader = DataLoader(
        synthetic_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    print(
        f"Processing {len(real_dataset)} real images and {len(synthetic_dataset)} synthetic images..."
    )

    # Initialize metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    # Use RadDino for feature extraction
    model, processor = load_RadDino_encoder()
    rad_dino_fe = RadDinoFeatureExtractor(model, processor)
    fid_raddino = FrechetInceptionDistance(feature=rad_dino_fe).to(device)

    kid = KernelInceptionDistance(subset_size=1000, feature=2048).to(device)
    inception_score = InceptionScore(feature=2048).to(device)

    # Process real images
    print("Processing real images...")
    for batch in tqdm(real_dataloader):

        batch = batch.to(device)
        # Scale images from [0, 1] to [0, 255] as expected by torchmetrics
        batch = (batch * 255).to(torch.uint8)

        # Update FID and KID with real images
        fid.update(batch, real=True)
        kid.update(batch, real=True)

        # Update FID with RadDino Features
        fid_raddino.update(batch, real=True)

    # Process synthetic images
    print("Processing synthetic images...")
    for batch in tqdm(synthetic_dataloader):

        batch = batch.to(device)
        batch = (batch * 255).to(torch.uint8)

        # Update FID and KID with synthetic images
        fid.update(batch, real=False)
        # Update FID with RadDino features
        fid_raddino.update(batch, real=False)

        kid.update(batch, real=False)

        # Update inception score
        inception_score.update(batch)

    # Calculate metrics
    print("Calculating metrics...")
    fid_value = fid.compute()
    fid_raddino_value = fid_raddino.compute()
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception_score.compute()

    print(f"FID: {fid_value.item()}")
    print(f"FID (RadDino): {fid_raddino_value.item()}")
    print(f"KID: {kid_mean.item()} ± {kid_std.item()}")
    print(f"Inception Score: {is_mean.item()} ± {is_std.item()}")

    # Save results
    results = {
        "FID": round(fid_value.item(), 3),
        "FID (RadDino)": round(fid_raddino_value.item(), 3),
        "KID": round(kid_mean.item(), 3),
        # 'KID (std)': kid_std.item(),
        "Inception Score": round(is_mean.item(), 3),
        # 'Inception Score (std)': is_std.item()
        "Extra Info": args.extra_info,
        "Alignment_score": np.nan,
    }

    print("RESULTS ... ")
    print(results)

    # Save to CSV
    results_df = pd.DataFrame([results])
    savename = "image_generation_metrics_debug.csv" if args.debug else "image_generation_metrics.csv"
    savepath = (
        os.path.join(args.results_savedir, savename)
        if args.results_savedir
        else savename
    )

    if os.path.exists(savepath):
        print("Appending to existing results file.")
        existing_df = pd.read_csv(savepath)
        results_row = list(results.values())

        existing_df.loc[len(results_df)] = results_row
        existing_df.to_csv(savepath, index=False)
    else:
        print("Creating new results file.")
        results_df = pd.DataFrame([results])
        results_df.to_csv(savepath, index=False)

    print("Results saved to ", savepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate FID, KID, and Inception Score for synthetic images."
    )
    parser.add_argument(
        "--synthetic_csv",
        type=str,
        required=True,
        help="CSV file containing paths to synthetic images.",
    )
    parser.add_argument(
        "--synthetic_img_col",
        type=str,
        default="img_savename",
        help="Col name in synthetic CSV for image paths.",
    )
    parser.add_argument(
        "--synthetic_img_dir", type=str, help="Directory containing synthetic images."
    )

    parser.add_argument(
        "--real_csv",
        type=str,
        required=True,
        help="CSV file containing paths to real images.",
    )
    parser.add_argument(
        "--real_img_col",
        type=str,
        default="path",
        help="Col name in real CSV for image paths.",
    )
    parser.add_argument(
        "--real_img_dir", type=str, help="Directory containing real images."
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing images."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--extra_info", type=str, default="Some AI Model", help="Extra info to link the results with the specific model."
    )

    parser.add_argument(
        "--results_savedir",
        type=str,
        default="Results",
        help="Directory to save the results.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    parser.add_argument(
        "--debug_samples", type=int, default=100, help="Debug Samples."
    )

    args = parser.parse_args()
    main(args)
