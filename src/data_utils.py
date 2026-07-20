from pathlib import Path

import pandas as pd

import hashlib

import imagehash

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "bonn_furniture_styles"
IMAGE_ROOT = DATASET_ROOT / "houzz"
SPLITS_ROOT = DATASET_ROOT / "splits"

SPLIT_FILES = {
    "train": SPLITS_ROOT / "train_split.txt",
    "validation": SPLITS_ROOT / "val_split.txt",
    "test": SPLITS_ROOT / "test_split.txt",
}

SELECTED_CATEGORIES = ("beds", "dressers")

def parse_split_file(split_path, split_name):
    """Parse one official Bonn dataset split file."""

    records = []

    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            left_text, marker, metadata_text = line.partition("METADATA:")

            left_fields = left_text.split()

            metadata_fields = [field.strip() for field in metadata_text.split(";")]

            if (not marker or len(left_fields) != 2 or len(metadata_fields) < 2):
                raise ValueError(
                    f"Malformed line {line_number} "
                    f"in {split_path.name}"
                )

            style, relative_path = left_fields
            product_subtype = metadata_fields[-2]
            metadata_style = metadata_fields[-1]

            relative_path = relative_path.replace("\\", "/")
            path_parts = Path(relative_path).parts

            if len(path_parts) < 4 or path_parts[0] != "houzz":
                raise ValueError(
                    f"Unexpected path on line {line_number}: "
                    f"{relative_path}"
                )
            category = path_parts[1]
            directory_style = path_parts[2]

            records.append(
                {
                    "split": split_name,
                    "relative_path": relative_path,
                    "category": category,
                    "style": style,
                    "directory_style": directory_style,
                    "product_subtype": product_subtype,
                    "metadata_style": metadata_style,
                    "metadata_attributes": ";".join(metadata_fields[:-2]),
                    "styles_match": (style.casefold() == directory_style.casefold() == metadata_style.casefold()),
                    "metadata_field_count": len(metadata_fields),

                }
            )

    return pd.DataFrame(records)


def load_dataset_splits():
    """Load the indipendent train, validation, and test splits."""

    split_dataframes = []

    for split_name, split_path in SPLIT_FILES.items():
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        split_dataframes.append(parse_split_file(split_path, split_name))

    return pd.concat(split_dataframes, ignore_index=True)


def inspect_image_file(relative_path, dataset_root=DATASET_ROOT):
    """Validate one image and collect properties used in later analyses"""

    relative_path = str(relative_path)
    image_path = dataset_root / relative_path

    result = {
        "relative_path":relative_path,
        "file_exists": image_path.is_file(),
        "image_readable": False,
        "width": None,
        "hight": None,
        "image_mode": None,
        "image_format": None,
        "file_size_bytes": None,
        "sha256": None,
        "perceptual_hash": None,
        "inspection_error": None,
    }

    if not result["file_exists"]:
        result["inspection_error"] = "File not found"
        return result
    
    try:
        result["file_size_bytes"] = image_path.stat().st_size

        sha256_hash = hashlib.sha256()

        with image_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                sha256_hash.update(chunk)

        result["sha256"] = sha256_hash.hexdigest()

        with Image.open(image_path) as image:
            image.load()

            result["width"], result["height"] = image.size
            result["image_mode"] = image.mode
            result["image_format"] = image.format
            result["perceptual_hash"] = str(imagehash.phash(image.convert("RGB")))

        result["image_readable"] = True

    except (OSError, ValueError, UnicodeDecodeError) as error:
        result["inspection_error"] = str(error)
    

    return result