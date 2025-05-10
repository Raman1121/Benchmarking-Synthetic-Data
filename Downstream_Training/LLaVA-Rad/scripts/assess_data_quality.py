import requests
import torch
from PIL import Image
from io import BytesIO
import os
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import ast
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

def get_labels_dict_from_string(x):
    return ast.literal_eval(x)

def parse_args():
    
    parser = argparse.ArgumentParser(description="Assess Data Quality")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the results.")
    parser.add_argument("--img_col", type=str, default='synthetic_filename', help="Column for caption in the metadata file.")
    parser.add_argument("--caption_col", type=str, default='annotated_prompt', help="Column for caption in the metadata file.")
    parser.add_argument("--labels_col", type=str, default='chexpert_labels', help="Column for labels in the metadata file.")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory where images are located.")
    parser.add_argument("--num_shards", type=str, default=None, help="Number of shards to divide the dataset into.")
    parser.add_argument("--shard", type=int, default=None, help="Shard ID.")
    
    return parser.parse_args()

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):

    META_PROMPT = "You are an expert radiologist and medical annotator. Your task is to assess the quality of an image given its description and classify the image as either 'High Quality', 'Medium Quality', or 'Low Quality'. Keep your responses limited to only these three options. If the image is not relevant to the description, respond with 'Not Relevant'." 

    disable_torch_init()

    model_path = "microsoft/llava-rad"
    model_base = "lmsys/vicuna-7b-v1.5"
    model_name = "llavarad"
    conv_mode = "v1"

    print("Loading pre-trained LLaVA-Rad Model")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    print("Done!")

    print("Loading Dataset...")
    df = pd.read_csv(args.metadata_csv)
    df[args.img_col] = df[args.img_col].apply(lambda x: os.path.join(args.image_dir, x))

    try:
        df[args.labels_col] = df[args.labels_col].apply(lambda x: get_labels_dict_from_string(x))
    except:
        pass

    if(args.num_shards is not None and args.shard is not None):
        all_shards = np.array_split(df, args.num_shards)
        df = all_shards[args.shard].reset_index(drop=True)
        print("Shard {} of {} loaded.".format(args.shard, args.num_shards))

    print("Loaded {} samples.".format(len(df)))
    print("Done!")

    ALL_RESPONSES = []

    start_time = time.time()

    # for i in range(len(df)):
    for i in tqdm(range(len(df))):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(df)}...")

        prompt = df[args.caption_col].iloc[i]
        image_file = df[args.img_col].iloc[i]
        image = load_image(image_file)

        query = META_PROMPT + "\n" + f"<image>\nClassify the image as either 'High Quality', 'Medium Quality', or 'Low Quality' given the prompt: {prompt}" + "\n"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = load_image(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].half().unsqueeze(0).cuda()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stopping_criteria = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        
        ALL_RESPONSES.append(outputs)

    df['Image Quality'] = ALL_RESPONSES
    print("Saving results to CSV...")

    if(args.num_shards is not None and args.shard is not None):
        filename = "metadata_with_quality_shard_{}.csv".format(args.shard)
    else:
        filename = "metadata_with_quality.csv"

    df.to_csv(os.path.join(args.output_dir, filename), index=False)
    print(f"Results saved to: {os.path.join(args.output_dir, filename)}")
    print("Done!")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 3600:.2f} hours")

if __name__ == "__main__":
    args = parse_args()
    main(args)
