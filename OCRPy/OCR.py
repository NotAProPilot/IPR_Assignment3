import pytesseract
from PIL import Image
import os


def recognize_handwriting(image_path):
    """Recognize handwriting from an image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to grayscale
        gray_img = img.convert("L")
        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(gray_img)
    return text


def main():
    # Example image path
    image_path = "OCRPy/images.jpeg"

    # Ensure the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Recognize handwriting from the given image
    recognized_text = recognize_handwriting(image_path)
    print("Recognized Text:")
    print(recognized_text)


if __name__ == "__main__":
    main()
