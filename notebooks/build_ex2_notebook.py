"""Generate ex2-customize-and-automate.ipynb.

Edit the cells here and rerun with `uv run python notebooks/build_ex2_notebook.py`.
The notebook is written without outputs, so it is safe to commit and to open in Colab.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HAFNIA_VERSION = "0.8.7"
PATH_NOTEBOOK = Path(__file__).parent / "ex2-customize-and-automate.ipynb"


def md(text: str):
    return new_markdown_cell(text.strip())


def code(text: str):
    return new_code_cell(text.strip())


cells = [
    md(
        """
# Exercise 2: Customize and automate with the Hafnia SDK and CLI

In Exercise 1 you launched a training run by clicking in the platform. In this notebook you do the same
from code, and you customize the data the model trains on:

1. **Set up:** connect the notebook to the Hafnia platform with your API key.
2. **CLI tour:** list datasets, trainers and environments from the command line.
3. **Datasets:** load the *sample* of a dataset and look inside it.
4. **Dataset recipes:** turn MNIST into a new "even vs. odd" task and save it as a recipe on the platform.
5. **Your own data:** build a small dataset of your own.
6. **Launch:** train the public classification trainer on your new recipe, with one CLI command.

**Time:** about 15 minutes. Run the cells from top to bottom with **Shift + Enter**.

> **In Google Colab:** this notebook runs on your own private Colab machine, so nobody else sees your work.
> - When you run the first cell, Colab warns that the notebook was not authored by Google. Click **Run anyway**.
> - Your changes are lost when you close the tab. To keep them, click **File → Save a copy in Drive**.
"""
    ),
    md(
        """
## 1. Set up

Install the Hafnia SDK and CLI. This takes about a minute in Colab.
"""
    ),
    code(
        f"""
%pip install -q hafnia=={HAFNIA_VERSION}
"""
    ),
    md(
        """
Enter **your name**. It goes into the name of everything you create, because all participants share the
`BreakoutSessionDevSummit` organization. Use letters, numbers and dashes, for example `jane-doe`.
"""
    ),
    code(
        """
NAME = "your-name"  # <-- change this, e.g. "jane-doe"
"""
    ),
    code(
        """
import os
import re

NAME = re.sub(r"[^a-z0-9-]+", "-", NAME.strip().lower()).strip("-")
assert NAME and NAME != "your-name", "Set NAME in the cell above to your own name"

os.environ["COLUMNS"] = "200"  # Wide enough for the CLI tables
print(f"Hello {NAME}!")
"""
    ),
    md(
        """
Now get your **API key**:

1. Log in at [hafnia.milestonesys.com](https://hafnia.milestonesys.com/) and check that
   `BreakoutSessionDevSummit` is your active organization.
2. Create an API key and copy it. We show you where during the session.
3. Run the cell below and paste the key into the box that appears.

The key is hidden while you type, and it is never printed in the notebook. Keep it private.
"""
    ),
    code(
        """
from getpass import getpass

API_KEY = getpass("Paste your Hafnia API key and press Enter: ")
assert API_KEY.startswith("ApiKey "), "That does not look like a Hafnia API key. Run this cell again and paste the full key, starting with 'ApiKey '"
"""
    ),
    md(
        """
Create a CLI profile with your key. One command configures both the CLI and the Python SDK.
"""
    ),
    code(
        """
!hafnia profile create "{API_KEY}" --name "breakout-session"
!hafnia profile ls
"""
    ),
    md(
        """
## 2. A quick tour of the CLI

Everything you clicked in Exercise 1 is also available from the `hafnia` command line. Lines that start
with `!` run as shell commands, exactly as they would in a terminal.
"""
    ),
    code(
        """
!hafnia --help
"""
    ),
    md(
        """
List the datasets that contain "mnist". 

"""
    ),
    code(
        """
!hafnia dataset ls --search mnist
"""
    ),
    md(
        """
List the public trainers (the same ones you picked from in Exercise 1) and the hardware you can train on.
"""
    ),
    code(
        """
!hafnia trainer ls --visibility public
!hafnia experiment environments
"""
    ),
    md(
        """
## 3. How a dataset works

`HafniaDataset.from_name` downloads the **sample** of a dataset (200 images for MNIST). Everything you do
with it here works the same way on the hidden dataset inside a training run.
"""
    ),
    code(
        """
from hafnia.dataset.hafnia_dataset import HafniaDataset

dataset = HafniaDataset.from_name("mnist", version="1.0.0")
dataset.print_stats()
"""
    ),
    md(
        """
A dataset has two parts:

- `dataset.info`: name, version and tasks (here, image classification with 10 classes).
- `dataset.samples`: one row per image, as a [Polars](https://pola.rs/) table with the annotations.
"""
    ),
    code(
        """
print(dataset.info.tasks[0].get_class_names())
dataset.samples.head(3)
"""
    ),
    md(
        """
Look at a few images. `Sample` gives typed access to one row, including the image and its labels.
"""
    ),
    code(
        """
import matplotlib.pyplot as plt
from hafnia.dataset.hafnia_dataset_types import Sample


def show_samples(dataset, n=8):
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for ax, sample_dict in zip(axes, dataset):
        sample = Sample(**sample_dict)
        ax.imshow(sample.draw_annotations(), cmap="gray")
        ax.set_title(sample.classifications[0].class_name)
        ax.axis("off")
    plt.show()


show_samples(dataset.shuffle(seed=42))
"""
    ),
    md(
        """
## 4. Dataset recipes: change the task without touching the data

A **dataset recipe** describes *how* to build a dataset: which datasets to use, which classes to keep,
merge or rename, and how many samples to select. The recipe is only a description, so it can run anywhere:
on the sample here, and on the full hidden dataset in the cloud.

Turn MNIST into an "even vs. odd" classifier. Map every digit to `even` or `odd`, and drop the nines.
"""
    ),
    code(
        """
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

class_mapping = {
    "0 - zero": "even",
    "1 - one": "odd",
    "2 - two": "even",
    "3 - three": "odd",
    "4 - four": "even",
    "5 - five": "odd",
    "6 - six": "even",
    "7 - seven": "odd",
    "8 - eight": "even",
    "9 - nine": "__REMOVE__",  # Drop all nines
}

recipe = DatasetRecipe.from_name("mnist", version="1.0.0").class_mapper(class_mapping=class_mapping)
print(recipe.as_json_str())
"""
    ),
    md(
        """
Nothing has run yet. Call `build()` to apply the recipe to the sample, and check the result before you use it
for training.
"""
    ),
    code(
        """
dataset_even_odd = recipe.build()
dataset_even_odd.print_class_distribution()
show_samples(dataset_even_odd.shuffle(seed=42))
"""
    ),
    md(
        """
The recipe works on the sample. Save it on the platform with your name, so you can train on it.
"""
    ),
    code(
        """
RECIPE_NAME = f"{NAME}-mnist-even-odd"

recipe.as_platform_recipe(recipe_name=RECIPE_NAME, overwrite=True)
!hafnia recipe ls --search "{NAME}"
"""
    ),
    md(
        """
Recipes can do much more, for example merge several datasets into one, select samples by class, or
create new train/validation/test splits:

```python
DatasetRecipe.from_merger(recipes=[recipe_a, recipe_b])
recipe.select_samples_by_class_name(name=["Person", "Vehicle"], task_name="object_detection")
recipe.splits_by_ratios({"train": 0.8, "val": 0.1, "test": 0.1})
```
"""
    ),
    md(
        """
## 5. Bring your own data

You can also add your own images. If your data is already in YOLO or COCO format, one line converts it:

```python
HafniaDataset.from_yolo_format(path)
HafniaDataset.from_coco_format(path)
```

For any other format, build the samples yourself. As an example, draw 60 images of circles, squares and
triangles, label each one, and turn them into a `HafniaDataset`.
"""
    ),
    code(
        """
import random
from pathlib import Path

from PIL import Image, ImageDraw

path_images = Path("data/my-shapes")
path_images.mkdir(parents=True, exist_ok=True)
shape_names = ["circle", "square", "triangle"]


def draw_shape(shape, size=64):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    color = tuple(random.randint(0, 200) for _ in range(3))
    x0, y0 = random.randint(4, 20), random.randint(4, 20)
    x1, y1 = x0 + random.randint(28, 40), y0 + random.randint(28, 40)
    if shape == "circle":
        draw.ellipse((x0, y0, x1, y1), fill=color)
    elif shape == "square":
        draw.rectangle((x0, y0, x1, y1), fill=color)
    else:
        draw.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=color)
    return image


random.seed(42)
my_images = []
for i in range(60):
    shape = shape_names[i % 3]
    path_image = path_images / f"{shape}_{i:02d}.png"
    draw_shape(shape).save(path_image)
    my_images.append((path_image, shape))

print(f"Created {len(my_images)} images in {path_images}")
"""
    ),
    md(
        """
Describe each image as a `Sample` with its label, and the dataset as a `DatasetInfo` with its task and classes.
"""
    ),
    code(
        """
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, TaskInfo
from hafnia.dataset.primitives import Classification

samples = []
for i, (path_image, shape) in enumerate(my_images):
    label = Classification(class_name=shape, class_idx=shape_names.index(shape))
    split = "train" if i < 48 else "validation"
    samples.append(Sample(file_path=str(path_image), height=IMAGE_SIZE, width=IMAGE_SIZE, split=split, classifications=[label]))

info = DatasetInfo(
    dataset_name=f"{NAME}-my-shapes",
    version="0.0.1",
    tasks=[TaskInfo.from_class_names(primitive=Classification, class_names=shape_names)],
)
my_dataset = HafniaDataset.from_samples_list(samples_list=samples, info=info)
my_dataset.print_stats()
show_samples(my_dataset.shuffle(seed=1))
"""
    ),
    md(
        """
Your images are now a `HafniaDataset`, in the same format as every dataset in the Hafnia data library. From
here you can:

- combine them with platform datasets in a recipe,
- write them to disk with `my_dataset.write(path)`, or export them with `to_yolo_format` / `to_coco_format`,
- upload them to your organization, so trainers can use them (see below).
"""
    ),
    md(
        """
### Uploading a dataset to the platform

When you have a dataset you want to train on, upload it with one call: `my_dataset.upload_to_platform()`.

- The dataset name and version come from the `DatasetInfo` (here `{NAME}-my-shapes`, version `0.0.1`).
- The images and annotations are synced to your organization's storage, and the dataset shows up in the
  **Data library**, where you can browse samples and class distributions.
- Uploading the same name and version again is refused by default (or asks you to confirm). Use
  `my_dataset.upload_to_platform(allow_version_overwrite=True)` to replace an existing version, or bump the
  `version` in `DatasetInfo` to keep both.
- Once uploaded, use the dataset like any other: by name in a recipe, or with `-d` in `hafnia experiment create`.

We skip the upload today, so that 50 participants don't fill the shared organization with test datasets.
Try it in your own organization after the session.
"""
    ),
    md(
        """
## 6. Launch training on your recipe

Launch the public **Image Classification Trainer** on your even/odd recipe. This is the same trainer as in
Exercise 1. Only the data changed, and you did not have to write any training code.

First, look up the trainer's ID.
"""
    ),
    code(
        """
from hafnia.platform.trainer_package import get_trainer_packages

public_trainers = get_trainer_packages(visibility="PUBLIC")
TRAINER_ID = next(t["id"] for t in public_trainers if "Image Classification" in t["name"])
print(f"Image Classification Trainer: {TRAINER_ID}")
"""
    ),
    md(
        """
Now launch the experiment. The options say *what* to run:

| Option | Value |
|---|---|
| `--name` | the experiment name, with your name in it |
| `--recipe` | your even/odd recipe |
| `--trainer-id` | the public classification trainer |
| `--cmd` | the script to run and its arguments, like the command builder in Exercise 1 |
| `--environment` | the hardware (`Lite` is one GPU) |
"""
    ),
    code(
        """
EXPERIMENT_NAME = f"{NAME}-mnist-even-odd"
TRAIN_CMD = "python scripts/train.py --epochs 3"

!hafnia experiment create --name "{EXPERIMENT_NAME}" --recipe "{RECIPE_NAME}" --trainer-id "{TRAINER_ID}" --cmd "{TRAIN_CMD}" --environment Lite
"""
    ),
    md(
        """
Your experiment is running on the full hidden MNIST dataset, with your even/odd recipe applied in the cloud.

Open [Training-aaS → Experiments](https://hafnia.milestonesys.com/dashboard/training-aas/experiments),
click **My Experiments** and find your experiment. You can also list it from here:
"""
    ),
    code(
        """
!hafnia experiment ls --search "{NAME}"
"""
    ),
    md(
        """
## Done! What's next?

You have:

- configured the Hafnia CLI and SDK with one command,
- explored a dataset through its downloadable sample,
- changed the training task with a dataset recipe, without touching the data or the trainer,
- built a dataset from your own images,
- launched a cloud training run on hidden data from a notebook.

**Want more?**

- Launch a second experiment with other arguments, for example `--epochs 5`, or try `scripts/benchmark.py`.
- Change the recipe, for example merge `mnist` with itself or select only some digits, and launch again.

**Build your own trainer.** A trainer is a folder with a `scripts/train.py` that reads a Hafnia dataset.
Develop it on your own machine against the sample dataset, then upload it with `hafnia trainer create` and
train it on the hidden dataset. Start from the public trainers:
[trainer-classification](https://github.com/milestone-hafnia/trainer-classification) and
[trainer-object-detection](https://github.com/milestone-hafnia/trainer-object-detection).
"""
    ),
]

notebook = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
)
nbformat.write(notebook, PATH_NOTEBOOK)
print(f"Wrote {PATH_NOTEBOOK}")
