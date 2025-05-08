import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import shutil

def combine_shards(args):
    """
    Each CSV file contains a single row.
    Create a new CSV file that contains the average of all the columns that are float values.
    """

    # List all CSV files in the shard directory
    csv_files = [f for f in os.listdir(args.shards_dir) if f.endswith('.csv')]
    

    FID = []
    KID = []
    Precision = []
    Recall = []
    Density = []
    Coverage = []
    
    for csv_file in csv_files:
        _df = pd.read_csv(os.path.join(args.shards_dir, csv_file))
        FID.append(_df['FID (DenseNet)'].values[0])
        KID.append(_df['KID (DenseNet)'].values[0])
        Precision.append(_df['Precision'].values[0])
        Recall.append(_df['Recall'].values[0])
        Density.append(_df['Density'].values[0])
        Coverage.append(_df['Coverage'].values[0])

    # Create a new DataFrame with the average of all the columns
    combined_df = pd.DataFrame({
        'FID': [round(sum(FID)/len(FID), 3)],
        'KID': [round(sum(KID)/len(KID), 3)],
        'Precision': [round(sum(Precision)/len(Precision), 3)],
        'Recall': [round(sum(Recall)/len(Recall), 3)],
        'Density': [round(sum(Density)/len(Density), 3)],
        'Coverage': [round(sum(Coverage)/len(Coverage), 3)],
        "Alignment_score": np.nan,
        "Extra Info": args.extra_info
    })
    
    return combined_df

def main(args):

    combined_df = combine_shards(args)
    savename = "image_generation_metrics_densenet.csv"
    savepath = os.path.join(args.output_dir, savename)

    # Read the existing CSV file if it exists
    if os.path.exists(savepath):
        existing_df = pd.read_csv(savepath)
        # Append the new row to the existing DataFrame
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    combined_df.to_csv(savepath, index=False)
    print("Saved to: ", savepath)

    if(args.delete_after_combining):
        # Remove all csv files in shards_dir
        csv_files = [f for f in os.listdir(args.shards_dir) if f.endswith('.csv')]
        for file in csv_files:
            os.remove(os.path.join(args.shards_dir, file))


def parse_args():
    parser = argparse.ArgumentParser(description="Combine shards of CSV files.")
    parser.add_argument(
        "--shards_dir",
        type=str,
        required=True,
        help="Directory containing the shard CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing the shard CSV files.",
    )
    parser.add_argument(
        "--extra_info",
        type=str,
        default="",
        help="Extra info to be added to the output CSV file.",
    )
    parser.add_argument(
        "--delete_after_combining",
        action="store_true",
        help="Delete the directory in which shards were stored after combining.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args)