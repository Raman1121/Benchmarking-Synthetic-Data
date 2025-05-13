import datasets
import pandas as pd
import os
import ast

# TODO: Add BibTeX citation
_CITATION = """\
@article{yourarticle,
  author    = {Your Name},
  title     = {My Awesome Image Collection with Quality Splits},
  journal   = {Your Journal},
  year      = {2024},
}
"""

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This dataset contains 200,000 images with associated metadata, primarily focusing on [describe your dataset's content and purpose].
The metadata includes filenames, image quality labels, and [other relevant information from your CSV].
The dataset is split into 'High Quality', 'Medium Quality', 'Low Quality' based on the 'Image Quality' column
in the metadata. An 'All' split (named 'train') containing all images is also provided as the default.
"""

_HOMEPAGE = "https://huggingface.co/datasets/raman07/SynthCheX-230K" # Replace with your dataset's HF Hub URL

_LICENSE = "apache-2.0" 

# Define the name of your image folder and metadata file
_IMAGE_FOLDER = "images"
_METADATA_FILE = "metadata_with_generations_subset.csv"
# Define the column name in the CSV that holds the quality label
_QUALITY_COLUMN = "Image Quality"


class SynthCheXDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0") # Incremented version for the change

    def _info(self):
        # Define the features of your dataset.
        # 'image' is a special feature for images.
        # For other columns from your CSV, specify their dtype.
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "file": datasets.Value("string"), # Assuming 'filename' is a column in your CSV
                    "prompt": datasets.Value("string"),   # Example: if you have a 'prompt' column
                    _QUALITY_COLUMN: datasets.Value("string"), # Include the quality column itself
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Path to the metadata file within the dataset repository
        metadata_path = os.path.join(dl_manager.manual_dir, _METADATA_FILE)
        # Path to the image folder within the dataset repository
        image_dir_path = os.path.join(dl_manager.manual_dir, _IMAGE_FOLDER)

        if not os.path.exists(metadata_path):
             raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
        if not os.path.isdir(image_dir_path):
             raise FileNotFoundError(f"Image folder not found at {image_dir_path}.")

        quality_splits = ["High Quality", "Medium Quality"]
        split_generators = []

        # Create generators for each specific quality split
        for quality_label in quality_splits:
            split_generators.append(
                datasets.SplitGenerator(
                    name=quality_label, # Name the split directly with the label
                    gen_kwargs={
                        "metadata_filepath": metadata_path,
                        "image_dir": image_dir_path,
                        "quality_filter": quality_label, # Pass the label to filter by
                    },
                )
            )

        # Create generator for the "All" split, naming it 'train' (standard default)
        split_generators.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_filepath": metadata_path,
                    "image_dir": image_dir_path,
                    "quality_filter": None,
                },
            )
        )

        return split_generators


    def _generate_examples(self, metadata_filepath, image_dir, quality_filter):
        """Yields examples, filtering by quality_filter if provided."""
        # Load your metadata
        df = pd.read_csv(metadata_filepath)

        # Iterate through your metadata and yield examples matching the quality_filter
        count = 0
        for idx, row in df.iterrows():
            current_quality = row[_QUALITY_COLUMN]

            # Apply filtering based on the quality_filter argument
            if quality_filter is not None and current_quality != quality_filter:
                continue # Skip row if it doesn't match the desired quality split

            # --- Row matches the filter (or filter is None for 'All'), proceed ---
            image_filename = row["synthetic_filename"]
            image_path = os.path.join(image_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_path} not found, skipping entry with id {idx}.")
                continue

            # Use a unique key for each example within its split generator
            # Simple count ensures uniqueness within this specific call to _generate_examples
            example_key = f"{quality_filter or 'all'}_{count}"
            count += 1

            # YOU MUST ADJUST THE DICTIONARY KEYS to match the `features` defined in `_info()`
            yield example_key, {
                "image": image_path,
                "file": image_filename,
                "prompt": row.get("annotated_prompt", ""),
                _QUALITY_COLUMN: current_quality, # Include the quality label in the example
            }