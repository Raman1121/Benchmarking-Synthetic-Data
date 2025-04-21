import os
import pandas as pd
import shutil
from tqdm import tqdm
import json

def write_text_to_file(text: str, filename: str) -> None:
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

CSV_PATH = '/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES'
IMG_DIR = "/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

train_csv = pd.read_csv(CSV_PATH+'LLAVARAD_ANNOTATIONS_TRAIN.csv')
test_csv = pd.read_csv(CSV_PATH+'LLAVARAD_ANNOTATIONS_TEST.csv')

train_csv['filename'] = train_csv['path'].apply(lambda x: x.split("/")[-1])
test_csv['filename'] = test_csv['path'].apply(lambda x: x.split("/")[-1])

SAVE_DIR_TRAIN = '/pvc/MIMIC_ARRANGED/Train/'
os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
SAVE_DIR_TEST = '/pvc/MIMIC_ARRANGED/Test/'
os.makedirs(SAVE_DIR_TEST, exist_ok=True)

print("!!!Copying the files")
for i in range(len(train_csv)):
    prompt = train_csv['annotated_prompt'][i]
    source_path = os.path.join(IMG_DIR, train_csv['path'][i])
    dest_path = os.path.join(SAVE_DIR_TRAIN, train_csv['filename'][i])

    # Move image
    shutil.copy(source_path, dest_path)

    # print(dest_path.replace(".jpg", '.txt'))
    write_text_to_file(prompt, dest_path.replace(".jpg", '.txt'))

print("Preparing the JSON file")
meta_data = {
    "name": "sana-dev",
    "__kind__": "Sana-ImgDataset",
    "img_names": []
}

filenames = train_csv['filename']
meta_data['img_names'].extend(filenames)

# Save
savename = SAVE_DIR_TRAIN + "meta_data.json"
with open(savename, 'w', encoding='utf-8') as file:
    json.dump(meta_data, file, indent=4)