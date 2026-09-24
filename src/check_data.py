"""Check image–mask pairs and the relationship between validation subsets.

This script reads files only. It verifies that ``small`` and ``large`` partition
the 65-image validation set and that their copies have identical byte content.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def digest(path: Path) -> str:
    """Hash a file in chunks so image size does not affect memory use."""
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def pairs(images: Path, masks: Path, expected: int) -> dict[str, tuple[Path, Path]]:
    """Match each JPEG to its PNG label by stem and check dimensions/classes."""
    image_files = {path.stem: path for path in images.glob("*.jpg")}
    mask_files = {path.stem: path for path in masks.glob("*.png")}
    assert len(image_files) == len(mask_files) == expected
    assert image_files.keys() == mask_files.keys()
    result = {}
    for stem, image_path in image_files.items():
        mask_path = mask_files[stem]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            assert image.size == mask.size, stem
            minimum, maximum = mask.getextrema()
            assert 0 <= minimum <= maximum <= 1, stem
        result[stem] = image_path, mask_path
    return result


def main() -> None:
    dataset = DATA / "slope_fracture"
    pairs(dataset / "img_dir/train", dataset / "ann_dir/train", 264)
    validation = pairs(dataset / "img_dir/val", dataset / "ann_dir/val", 65)
    groups = {}
    for name, count in (("small", 40), ("large", 25)):
        groups[name] = pairs(DATA / name / "img", DATA / name / "ann", count)
        for stem, (image, mask) in groups[name].items():
            val_image, val_mask = validation[stem]
            assert digest(image) == digest(val_image), (name, stem, "image")
            assert digest(mask) == digest(val_mask), (name, stem, "mask")
    assert not (groups["small"].keys() & groups["large"].keys())
    assert groups["small"].keys() | groups["large"].keys() == validation.keys()

    examples = DATA / "full_views"
    for stem in ("a11", "a13", "a14"):
        with Image.open(examples / f"{stem}.png") as image, Image.open(
            examples / f"{stem}_mask.png"
        ) as mask:
            assert image.size == mask.size == (2560, 1440), stem
            minimum, maximum = mask.getextrema()
            assert 0 <= minimum <= maximum <= 255, stem
        assert (examples / f"{stem}.json").is_file(), stem
    print("Data check passed: 264 training, 65 validation, 40 small, 25 large, 3 full views.")


if __name__ == "__main__":
    main()
