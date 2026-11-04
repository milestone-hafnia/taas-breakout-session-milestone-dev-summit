from pathlib import Path
from typing import List

import numpy as np
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import Bbox, Bitmask, Classification
from hafnia.dataset.primitives.polygon import Polygon
from PIL import Image
from rich import print as rprint

##### SETUP #####
path_tmp = Path(".data/tmp")
path_tmp.mkdir(parents=True, exist_ok=True)


#### START #####
# Load sample dataset
dataset = HafniaDataset.from_name("mnist", version="1.0.0")
dataset.print_stats()

# Get access to a few full and public dataset through Hafnia (no login required)
# Available datasets: "mnist", "caltech-101", "caltech-256", "cifar10", "cifar100", "coco-2017"
public_dataset = HafniaDataset.from_name_public_dataset("mnist", n_samples=100)
public_dataset.print_stats()


#### Operations on datasets #####
# Create a dataset split for training
dataset_train = dataset.create_split_dataset("train")

small_dataset = dataset.select_samples(n_samples=10, seed=42)  # Selects 10 samples from the dataset
shuffled_dataset = dataset.shuffle(seed=42)  # Shuffle the dataset

# Create dataset splits by ratios
split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
new_dataset_splits = dataset.splits_by_ratios(split_ratios)

# Get only samples with specific class names
dataset_ones = dataset.select_samples_by_class_name(name="1 - one", primitive=Classification)


# Rename class names with mapping
class_mapping_strict = {
    "0 - zero": "even",
    "1 - one": "odd",
    "2 - two": "even",
    "3 - three": "odd",
    "4 - four": "even",
    "5 - five": "odd",
    "6 - six": "even",
    "7 - seven": "odd",
    "8 - eight": "even",
    "9 - nine": "odd",
}
dataset_mapped = dataset.class_mapper(class_mapping=class_mapping_strict)
dataset_mapped.print_class_distribution()

# Support Chaining Operations (load, shuffle, select samples)
dataset = (
    HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0").shuffle(seed=42).select_samples(n_samples=10)
)

## What is a hafnia dataset? ###
# Dataset information is stored in 'dataset.info'
rprint(dataset.info)

# Annotations are stored in 'dataset.samples' as a Polars DataFrame
dataset.samples.head(2)

## Custom dataset operations and statistics ##
# Want custom dataset transformations or statistics? Use the polars table (dataset.samples) directly
n_objects = dataset.samples["bboxes"].list.len().sum()
n_objects = dataset.samples[Bbox.column_name()].list.len().sum()  # Use Bbox.column_name() to avoid magic variables
n_classifications = dataset.samples[Classification.column_name()].list.len().sum()

class_counts = dataset.samples[Classification.column_name()].explode().struct.field("class_name").value_counts()
class_counts = dataset.samples[Bbox.column_name()].explode().struct.field("class_name").value_counts()
rprint(dict(class_counts.iter_rows()))

# Access the first sample in the training split - data is stored in a dictionary
sample_dict = dataset[0]

# Dataset can also be iterated to get sample data
for sample_dict in dataset:
    break

# Unpack dict into a Sample-object! Important for data validation, useability, IDE completion and mypy hints
sample: Sample = Sample(**sample_dict)

bboxes: List[Bbox] = sample.bboxes  # Use 'sample.bboxes' access bounding boxes as a list of Bbox objects
bitmasks: List[Bitmask] = sample.bitmasks  # Use 'sample.bitmasks' to access bitmasks as a list of Bitmask objects
polygons: List[Polygon] = sample.polygons  # Use 'sample.polygons' to access polygons as a list of Polygon objects
classifications: List[Classification] = sample.classifications  # As a list of Classification objects


# Read image using the sample object
image: np.ndarray = sample.read_image()

# Visualize sample and annotations
image_with_annotations = sample.draw_annotations()

# Save the image with annotations to a temporary directory

Image.fromarray(image_with_annotations).save(path_tmp / "sample_with_annotations.png")
