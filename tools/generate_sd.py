import argparse
import random
import os
import pandas as pd
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.state import AcceleratorState

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pipeline(model_path, device):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        safety_checker=None,
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    return pipe

class DummyDataset(Dataset):
    def __init__(self, data_list):
    
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]

def main(args):

    assert args.model_path is not None

    COUNT=0
    ALL_FILENAMES = []
    ALL_PROMPTS = []

    prompt_info_df = pd.DataFrame()

    if("v1_4" in args.model_path):
        args.save_dir = os.path.join(args.save_dir, "SD_v1_4")
    elif("v1_5" in args.model_path):
        args.save_dir = os.path.join(args.save_dir, "SD_v1_5")
    elif("v2" in args.model_path):
        args.save_dir = os.path.join(args.save_dir, "SD_v2")

    if(args.extra_info is not None):
        args.save_dir = args.save_dir + "_" + args.extra_info

    os.makedirs(args.save_dir, exist_ok=True)
    print("Saving images to {}".format(args.save_dir))

    seed_everything(seed=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(args.model_path, device)

    accelerator = Accelerator()

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

    print("Generating Images...")
    for batch in tqdm(dataloader):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)
        
        with autocast_ctx:
            outputs = pipe(
                batch,
                num_inference_steps=50,
                guidance_scale=7.5,
                num_images_per_prompt=1,
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
    parser = argparse.ArgumentParser(description="diffusion inference")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_prompts_path", default="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--savedir", type=str, default="/pvc/SYNTHETIC_IMAGES/")
    parser.add_argument("--extra_info", type=str, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    args = parser.parse_args()

    main(args)