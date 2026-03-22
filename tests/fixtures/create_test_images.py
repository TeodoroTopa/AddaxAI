"""Generate synthetic test images for fixture set.

Creates simple test images using Pillow for testing postprocessing
and export functionality. No external models required.
"""

import os
from PIL import Image, ImageDraw


def create_test_images() -> None:
    """Generate 5 synthetic test images in the fixtures/images directory."""
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(fixtures_dir, "images")

    # Create image 1: Simple animal (orange shape on blue background)
    img1 = Image.new("RGB", (400, 300), color="lightblue")
    draw = ImageDraw.Draw(img1)
    draw.rectangle([100, 100, 300, 250], fill="orange", outline="darkorange", width=3)
    draw.text((150, 150), "Animal", fill="black")
    img1.save(os.path.join(images_dir, "test_animal.jpg"), quality=85)

    # Create image 2: Empty scene (no detections)
    img2 = Image.new("RGB", (400, 300), color="lightgreen")
    draw = ImageDraw.Draw(img2)
    draw.text((150, 130), "Empty Scene", fill="darkgreen")
    img2.save(os.path.join(images_dir, "test_empty.jpg"), quality=85)

    # Create image 3: Person (red shape on gray background)
    img3 = Image.new("RGB", (400, 300), color="lightgray")
    draw = ImageDraw.Draw(img3)
    draw.rectangle([80, 50, 320, 280], fill="red", outline="darkred", width=3)
    draw.text((150, 150), "Person", fill="white")
    img3.save(os.path.join(images_dir, "test_person.jpg"), quality=85)

    # Create image 4: Vehicle (yellow shape on dark background)
    img4 = Image.new("RGB", (400, 300), color="dimgray")
    draw = ImageDraw.Draw(img4)
    draw.rectangle([120, 120, 280, 220], fill="yellow", outline="orange", width=3)
    draw.text((150, 160), "Vehicle", fill="black")
    img4.save(os.path.join(images_dir, "test_vehicle.jpg"), quality=85)

    # Create image 5: Multiple detections
    img5 = Image.new("RGB", (400, 300), color="lavender")
    draw = ImageDraw.Draw(img5)
    draw.rectangle([20, 20, 120, 120], fill="blue", outline="darkblue", width=2)
    draw.text((40, 60), "Det 1", fill="white")
    draw.rectangle([200, 150, 380, 280], fill="purple", outline="indigo", width=2)
    draw.text((240, 200), "Det 2", fill="white")
    img5.save(os.path.join(images_dir, "test_multi.jpg"), quality=85)

    print(f"Created 5 synthetic test images in {images_dir}/")


if __name__ == "__main__":
    create_test_images()
