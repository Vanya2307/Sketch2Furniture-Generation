import pytest
from PIL import Image

from src.data_utils import (
    SELECTED_CATEGORIES,
    SPLIT_FILES,
    inspect_image_file,
    load_dataset_splits,
    parse_split_file,
)


VALID_LINE = (
    "Asian houzz/beds/Asian/19726asian-daybeds.jpg "
    "METADATA:beds;Asian"
)


def write_split_file(directory, lines):
    split_path = directory / "split.txt"
    split_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return split_path


def test_parse_split_file_extracts_expected_fields(tmp_path):
    split_path = write_split_file(tmp_path, [VALID_LINE])

    records = parse_split_file(split_path, "train")

    assert len(records) == 1

    record = records.iloc[0]
    assert record["split"] == "train"
    assert record["category"] == "beds"
    assert record["style"] == "Asian"
    assert record["directory_style"] == "Asian"
    assert record["product_subtype"] == "beds"
    assert record["metadata_style"] == "Asian"
    assert record["metadata_field_count"] == 2
    assert record["styles_match"]


def test_parse_split_file_skips_blank_lines(tmp_path):
    split_path = write_split_file(
        tmp_path,
        ["", VALID_LINE, "   ", VALID_LINE],
    )

    assert len(parse_split_file(split_path, "train")) == 2


def test_parse_split_file_keeps_optional_attributes(tmp_path):
    line = (
        "Traditional "
        "houzz/dressers/Traditional/3634traditional.jpg "
        "METADATA:Picket House;W 52 in;Dressers;Traditional"
    )
    split_path = write_split_file(tmp_path, [line])

    record = parse_split_file(split_path, "test").iloc[0]

    assert record["category"] == "dressers"
    assert record["product_subtype"] == "Dressers"
    assert record["metadata_style"] == "Traditional"
    assert record["metadata_attributes"] == "Picket House;W 52 in"
    assert record["metadata_field_count"] == 4
    assert record["styles_match"]


def test_parse_split_file_detects_style_mismatch(tmp_path):
    line = (
        "Modern houzz/beds/Asian/19726asian-daybeds.jpg "
        "METADATA:beds;Asian"
    )
    split_path = write_split_file(tmp_path, [line])

    assert not parse_split_file(split_path, "train").iloc[0]["styles_match"]


@pytest.mark.parametrize(
    "line",
    [
        "Asian houzz/beds/Asian/image.jpg",
        "Asian METADATA:beds;Asian",
        "Asian houzz/beds/Asian/image.jpg METADATA:beds",
        "Asian beds/Asian/image.jpg METADATA:beds;Asian",
        "Asian houzz/beds/image.jpg METADATA:beds;Asian",
    ],
)
def test_parse_split_file_rejects_malformed_lines(tmp_path, line):
    split_path = write_split_file(tmp_path, [line])

    with pytest.raises(ValueError):
        parse_split_file(split_path, "train")


def test_inspect_image_file_reports_valid_image(tmp_path):
    image_directory = tmp_path / "houzz" / "beds"
    image_directory.mkdir(parents=True)
    Image.new("RGB", (350, 350), "white").save(
        image_directory / "sample.png"
    )

    result = inspect_image_file(
        "houzz/beds/sample.png",
        dataset_root=tmp_path,
    )

    assert result["file_exists"]
    assert result["image_readable"]
    assert (result["width"], result["height"]) == (350, 350)
    assert result["image_mode"] == "RGB"
    assert result["image_format"] == "PNG"
    assert len(result["sha256"]) == 64
    assert result["perceptual_hash"] is not None
    assert result["inspection_error"] is None


def test_inspect_image_file_reports_missing_file(tmp_path):
    result = inspect_image_file(
        "houzz/beds/absent.png",
        dataset_root=tmp_path,
    )

    assert not result["file_exists"]
    assert not result["image_readable"]
    assert result["inspection_error"] == "File not found"


def test_inspect_image_file_reports_unreadable_file(tmp_path):
    broken_path = tmp_path / "broken.jpg"
    broken_path.write_text("not an image", encoding="utf-8")

    result = inspect_image_file("broken.jpg", dataset_root=tmp_path)

    assert result["file_exists"]
    assert not result["image_readable"]
    assert result["inspection_error"] is not None
    assert result["sha256"] is not None


def test_selected_categories():
    assert SELECTED_CATEGORIES == ("beds", "dressers")


@pytest.mark.skipif(
    not all(path.is_file() for path in SPLIT_FILES.values()),
    reason="Bonn dataset split files are not available",
)
def test_load_dataset_splits_returns_all_splits():
    splits = load_dataset_splits()

    assert set(splits["split"]) == {"train", "validation", "test"}
    assert not splits["relative_path"].isna().any()