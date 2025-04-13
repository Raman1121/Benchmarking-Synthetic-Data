"""
We calculate the image-text aignment score using the
health_multimodal toolbox (https://github.com/microsoft/hi-ml)
"""

from health_multimodal.text import get_bert_inference
from health_multimodal.image import get_image_inference
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
import warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Image-Text Alignment Score b/w Real Prompt and Synthetic Images"
    )
    parser.add_argument(
        "--synthetic_csv",
        type=str,
        default=None,
        help="CSV File containing the path to synthetic images and corresponding prompts.",
    )
    parser.add_argument(
        "--synthetic_img_dir",
        type=Path,
        default=None,
        help="Directory containing the synthetic images.",
    )

    parser.add_argument(
        "--synthetic_prompts_col",
        type=str,
        default="prompt",
        help="Column name in the CSV file containing the prompts.",
    )
    parser.add_argument(
        "--synthetic_img_col",
        type=str,
        default="img_savename",
        help="Column name in the CSV file containing the image paths.",
    )

    parser.add_argument(
        "--results_savedir",
        type=str,
        default="Results",
        help="Directory to save the results.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--extra_info", type=str, default="Some AI Model", help="Extra info to link the results with the specific model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    parser.add_argument(
        "--debug_samples", type=int, default=100, help="Debug Samples."
    )

    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    if args.debug:
        warnings.filterwarnings("ignore")
        print("Debug mode is ON. Make sure this behavior is intended.")

    text_inference = get_bert_inference()
    image_inference = get_image_inference()
    image_text_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )

    prompts_df = pd.read_csv(args.synthetic_csv)

    if args.debug:
        prompts_df = prompts_df.sample(n=args.debug_samples, random_state=42).reset_index(drop=True)

    prompts_df[args.synthetic_img_col] = prompts_df[args.synthetic_img_col].apply(
        lambda x: args.synthetic_img_dir.joinpath(x)
    )
    synthetic_img_paths = prompts_df[args.synthetic_img_col].tolist()
    prompts = prompts_df[args.synthetic_prompts_col].tolist()

    print("Length of the dataset: ", len(prompts_df))

    all_scores = []

    for i in tqdm(range(len(prompts))):
        img_path = synthetic_img_paths[i]
        text = prompts[i]
        _score = round(
            image_text_inference.get_similarity_score_from_raw_data(img_path, text), 3
        )
        all_scores.append(_score)
    mean_alignment_scores = round(np.mean(all_scores, axis=0), 3)

    print("RESULTS...")
    print("Mean Img-Text Alignment Score: ", mean_alignment_scores)

    savename = "image_generation_metrics_debug.csv" if args.debug else "image_generation_metrics.csv"
    # savename = "image_generation_metrics_debug.csv"
    savepath = os.path.join(args.results_savedir, savename)

    # Try to read if the dataframe already exists
    if os.path.exists(savepath):

        print("Appending to existing results file found at ", savepath)
        results_df = pd.read_csv(savepath)
        results_df["Alignment_score"] = mean_alignment_scores
        results_df["Extra Info"] = args.extra_info
        results_df.to_csv(savepath, index=False)
        print("Image-Text Alignment Scores saved to : ", savepath)
    else:
        # print("Creating new results file.")
        # results_df = pd.DataFrame(columns=["Alignment_score", "Extra Info"])
        # results_df["Alignment_score"] = mean_alignment_scores
        # results_df["Extra Info"] = args.extra_info
        print("ERROR!! Results CSV not found!!")
    
    


if __name__ == "__main__":
    args = parse_args()

    # Create the results directory if it doesn't exist
    os.makedirs(args.results_savedir, exist_ok=True)

    # Set random seed for reproducibility
    seed_everything(42)

    main(args)
