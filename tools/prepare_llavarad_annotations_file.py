import os
import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLaVA-Rad Annotations File")
    parser.add_argument(
        "--llavarad_json_file",
        type=str,
        default="MIMIC_Splits/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json",
        help="Path to the Original JSON file for LLaVA-Rad",
    )
    parser.add_argument(
        "--prompt_info_file",
        type=str,
        default="assets/CSV/prompt_INFO.csv",
        help="Path to the CSV file containing the path to synthetic image and corresponding prompt.",
    )
    
    return parser.parse_args()

def append_new_conversation(prompt):
    RESPONSE_TEMPLATE_LIST = []

    D1 = {'from': 'human',
    'value': '<image>\nDescribe the findings of the chest x-ray.\n'}

    D2 = {
        'from': 'gpt',
        'value': prompt
        }
    RESPONSE_TEMPLATE_LIST.append(D1)
    RESPONSE_TEMPLATE_LIST.append(D2)  
    
    return RESPONSE_TEMPLATE_LIST

def main(args):
    df = pd.read_json(args.llavarad_json_file)
    df_syn = pd.read_csv(args.prompt_info_file)

    ## Label each image as 'real' and 'synthetic'
    df['img_type'] = 'real'
    df_syn['img_type'] = 'synthetic'

    df_syn.rename(columns={'img_savename': 'image'}, inplace=True)

    # Convert each synthetic prompt into the format of LLaVA-Rad conversations
    df_syn['conversations'] = df_syn['prompt'].apply(lambda x: append_new_conversation(x))

    # Augment the original training data with synthetic data
    df_combined = pd.concat([df, df_syn], ignore_index=True)

    # df_combined.to_csv("MIMIC_Splits/chat_train_MIMIC_CXR_real_and_syn.csv", index=False)
    # Save the combined DataFrame to a json file
    df_combined.to_json("MIMIC_Splits/chat_train_MIMIC_CXR_real_and_syn.json", orient="records")
    print("Combined DataFrame saved to: MIMIC_Splits/chat_train_MIMIC_CXR_real_and_syn.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)