import torch
from diffusers import AutoPipelineForText2Image
import os

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

    # --- Configuration ---
    # 1. Base Model ID (Confirm this is the correct one you fine-tuned FROM)
    base_model_id = "stabilityai/stable-diffusion-3.5-medium"

    # 2. Path to the directory containing your LoRA weights
    lora_weights_dir = "/pvc/ai-toolkit/output/sd3-5_medium_lora/" # The directory you specified

    # 3. The specific filename of your LoRA weights within that directory
    lora_weights_filename = "sd3-5_medium_lora.safetensors" # The filename you specified

    # 4. Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32 # Use float16 for efficiency on GPU

    # --- Check if LoRA file exists ---
    lora_weights_path = os.path.join(lora_weights_dir, lora_weights_filename)
    if not os.path.exists(lora_weights_path):
        raise FileNotFoundError(f"LoRA weights file not found at: {lora_weights_path}")

    # --- Load the Base Pipeline ---
    print(f"Loading base pipeline: {base_model_id}")
    pipeline = AutoPipelineForText2Image.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16).to(device)

    print("Base pipeline loaded.")

    # --- Load the LoRA Weights ---
    # You provide the DIRECTORY where the weights are stored,
    # and specify the FILENAME using the `weight_name` argument.
    print(f"Loading LoRA weights from: {lora_weights_path}")
    pipeline.load_lora_weights(
        lora_weights_dir,              # Directory path
        weight_name=lora_weights_filename, # Specific filename
        adapter_name="sd3_medium_finetune_MIMIC" # Optional: Give your LoRA adapter a name
    )
    print("LoRA weights loaded successfully.")

    # --- Optional: Fuse LoRA (Consumes more memory, potential speedup, static) ---
    # If you don't need to dynamically change LoRA scale or unload it, you can fuse.
    # print("Fusing LoRA weights...")
    # pipeline.fuse_lora()
    # print("LoRA weights fused.")

    # --- Inference with LoRA ---
    # Note: For LoRA, you often control its influence using `cross_attention_kwargs`
    
    generator = torch.Generator(device=device).manual_seed(42)

    # Control LoRA influence (0.0 = no effect, 1.0 = full effect)
    lora_scale = 0.8 # Adjust this value (0.0 to 1.0 typically)

    print(f"Generating image with LoRA scale: {lora_scale}...")
    for batch in tqdm(dataloader):
        outputs = pipeline(
                batch,
                num_inference_steps=40,
                guidance_scale=4.5,
                num_images_per_prompt=1,
                max_sequence_length=512,
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
    
    try:
        prompt_info_df['prompt'] = ALL_PROMPTS
        prompt_info_df['img_savename'] = ALL_FILENAMES
        filename = "prompt_INFO.csv"
        prompt_info_df.to_csv(os.path.join(args.savedir, filename), index=False)
    except:
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion inference")
    parser.add_argument("--test_prompts_path", default="pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--savedir", default="/pvc/ai-toolkit/output/sd3-5_medium_lora/generated_images/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    args = parser.parse_args()

    main(args)