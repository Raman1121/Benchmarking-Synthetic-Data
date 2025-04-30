import torch
import os

import argparse
import random
import os
import pandas as pd
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
import torch
from diffusers import SanaPipeline
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.state import AcceleratorState

def load_sana_pipeline(model_path, device):
    pipe = SanaPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    return pipe

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DummyDataset(Dataset):
    def __init__(self, data_list):
    
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]
    
def main(args):

    prompt_info_df = pd.DataFrame()
    seed_everything(seed=42)

    COUNT=0
    ALL_FILENAMES = []
    ALL_PROMPTS = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.test_prompts_path is not None
    MIMIC_TEST_PROMPTS_PATH = args.test_prompts_path

    with open(MIMIC_TEST_PROMPTS_PATH, 'r') as file:
        test_prompts_list = file.readlines()

    test_prompts_list = [prompt.strip("\n") for prompt in test_prompts_list]

    if(args.debug):
        test_prompts_list = test_prompts_list[:6]

    print("{} Test Prompts found".format(len(test_prompts_list)))

    dataset = DummyDataset(test_prompts_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    os.makedirs(args.savedir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(42)

    pipeline = load_sana_pipeline(args.model_path, device)

    for batch in tqdm(dataloader):
        outputs = pipeline(
            num_inference_steps = 20,
            guidance_scale = 4.5,
            generator=generator
            )
        ALL_PROMPTS.extend(batch)

        images = outputs.images
        for idx, image in enumerate(images):
            idx += COUNT
            filename = "Prompt_{}.png".format(idx)
            savepath = os.path.join(args.savedir, filename)
            image.save(savepath)

            ALL_FILENAMES.append(filename)
        
        COUNT += args.batch_size
    
    prompt_info_df['prompt'] = ALL_PROMPTS
    prompt_info_df['img_savename'] = ALL_FILENAMES
    filename = "prompt_INFO.csv"
    prompt_info_df.to_csv(os.path.join(args.savedir, filename), index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sana inference")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_prompts_path", default="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt")
    # parser.add_argument("--test_prompts_path", default="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--savedir", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    args = parser.parse_args()

    main(args)

