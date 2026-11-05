from hafnia import utils
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset

COCO_VERSION = "1.0.0"
MIDWEST_VERSION = "1.0.0"
MNIST_VERSION = "1.0.0"

### Introducing DatasetRecipe ###
# A DatasetRecipe is a recipe for the dataset you want to create.
# The recipe itself is not executed - this is just a specification of the dataset you want!

# The 'DatasetRecipe' interface is similar to the 'HafniaDataset' interface.
# To demonstrate, we will first create a dataset with the regular 'HafniaDataset' interface.
# This line will get the "mnist" dataset, shuffle it, and select 20 samples.
dataset = HafniaDataset.from_name(name="mnist", version=MNIST_VERSION).shuffle().select_samples(n_samples=20)

# Now the same dataset is created using the 'DatasetRecipe' interface.
dataset_recipe = DatasetRecipe.from_name(name="mnist", version=MNIST_VERSION).shuffle().select_samples(n_samples=20)
dataset = dataset_recipe.build()
# Note that the interface is similar, but to actually create the dataset you need to call `build()` on the recipe.

# The recipe also allows you to combine multiple datasets and transformations that can be
# executed in the TaaS platform. This is demonstrated below:
if utils.is_hafnia_configured():  # First ensure you are connected to the hafnia platform
    # Upload the dataset recipe - this will make it available for TaaS and for users of your organization
    dataset_recipe.as_platform_recipe(recipe_name="example-mnist-recipe", overwrite=True)
