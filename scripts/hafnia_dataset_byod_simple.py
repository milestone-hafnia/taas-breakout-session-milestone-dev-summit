from pathlib import Path

from hafnia.dataset.hafnia_dataset import HafniaDataset

if __name__ == "__main__":
    path_data = Path(".data/PeterAndFriends/yolo_format")

    # From yolo format - use HafniaDataset.from_yolo_format() method
    user_dataset = HafniaDataset.from_yolo_format(path_data)

    user_dataset.info.dataset_name = "peter-and-friends"
    user_dataset.info.dataset_title = "Peter and Friends"
    user_dataset.print_stats()

    user_dataset.upload_to_platform(
        allow_version_overwrite=True,
        interactive=False,
        gallery_samples=user_dataset,
    )

    user_dataset.delete_on_platform(interactive=False)
    print("Done!")
