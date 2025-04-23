import argparse
import random
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_radedit_pipeline(device):
    # 1. UNet
    unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet")

    # 2. VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    # 3. Text encoder and tokenizer
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        model_max_length=128,
        trust_remote_code=True,
    )

    # 4. Scheduler
    scheduler = DDIMScheduler(
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        steps_offset=1,
    )

    # 5. Pipeline
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    )

    pipe = pipe.to(device)

    return pipe

class DummyDataset(Dataset):
    def __init__(self, data_list):
    
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]

def main(args):

    COUNT=0
    ALL_FILENAMES = []
    ALL_PROMPTS = []

    prompt_info_df = pd.DataFrame()

    seed_everything(seed=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_radedit_pipeline(device)

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
        outputs = pipe(
            batch,
            num_inference_steps=100,
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
    prompt_info_df.to_csv(os.path.join(args.savedir, "promtp_INFO_RadEdit.csv"), index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation of synthetic images using RadEdit")
    parser.add_argument("--test_prompts_path", default="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--savedir", default="/pvc/SYNTHETIC_IMAGES/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode to run on a small subset of data.",
    )
    args = parser.parse_args()

    main(args)