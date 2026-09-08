import json
from pathlib import Path

import requests


API_URL = "http://localhost:8000/predict"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def first_image(directory: Path) -> Path:
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image found in {directory}")
    return images[0]


def send_image(image_path: Path) -> dict:
    with image_path.open("rb") as image_file:
        response = requests.post(
            API_URL,
            files={"file": (image_path.name, image_file)},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def main():
    project_dir = Path(__file__).resolve().parent

    no_spill_image = first_image(project_dir / "data" / "Class_0")
    no_spill_response = send_image(no_spill_image)
    print("Class_0 (expected: no spill):")
    print(json.dumps(no_spill_response, indent=2))

    spill_image = first_image(project_dir / "data" / "Class_1")
    spill_response = send_image(spill_image)
    print("\nClass_1 (expected: spill):")
    print(json.dumps(spill_response, indent=2))


if __name__ == "__main__":
    main()
