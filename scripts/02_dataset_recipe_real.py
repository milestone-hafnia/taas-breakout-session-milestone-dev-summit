from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

if __name__ == "__main__":
    # Use case: Create Person and Vehicle datasets by combining "coco-2017" and "midwest-detection-traffic" datasets
    recipe_person_vehicle = DatasetRecipe.from_merge(
        recipe0=(
            DatasetRecipe.from_name(
                name="coco-2017",
                version="1.0.0",
            ).class_mapper(
                method="remove_undefined",
                task_name="object_detection",
                class_mapping=[
                    ("person", "Person"),
                    ("car", "Vehicle"),
                    ("truck", "Vehicle"),
                    ("motorcycle", "Vehicle"),
                    ("bicycle", "Vehicle"),
                    ("bus", "Vehicle"),
                    ("train", "Vehicle"),
                ],
            )
        ),
        recipe1=(
            DatasetRecipe.from_name(
                name="midwest-detection-traffic",
                version="1.0.0",
            ).class_mapper(
                method="strict",
                task_name="object_detection",
                class_mapping=[("Person*", "Person"), ("Vehicle.*", "Vehicle")],
            )
        ),
    ).select_samples_by_class_name(
        name=["Person", "Vehicle"],
    )

    dataset = recipe_person_vehicle.build()  # Check if it actually would work!?
    dataset.print_stats()  # Print dataset statistics

    recipe_person_vehicle.as_platform_recipe(recipe_name="person-vehicle-dataset", overwrite=True)

    recipe_person_vehicle_small = recipe_person_vehicle.select_samples(n_samples=1000, seed=42)
    recipe_person_vehicle_small.as_platform_recipe(recipe_name="person-vehicle-dataset-small", overwrite=True)
