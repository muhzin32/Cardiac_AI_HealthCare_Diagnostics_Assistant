from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image
import os


def extract_ecg_from_pdf(pdf_path, output_dir="uploads"):
    """
    Extracts the last page of a PDF, saves it as an image, 
    and crops the ECG region from it.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory where images should be saved.

    Returns:
        str: Path of the saved ECG image.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    save_last_page = os.path.join(output_dir, "last_page.png")
    save_ecg_image = os.path.join(output_dir, "ecg_image.png")

    # Get the total number of pages in the PDF
    pdf_reader = PdfReader(pdf_path)
    total_pages = len(pdf_reader.pages)

    # Convert only the last page to an image
    images = convert_from_path(pdf_path, first_page=total_pages, last_page=total_pages)

    if not images:
        print("Failed to extract the last page.")
        return None

    # Save the last page as an image
    last_page = images[0]
    last_page.save(save_last_page, "PNG")
    print(f"Last page saved at: {save_last_page}")

    # Get the actual dimensions of the image
    img_width, img_height = last_page.size

    # Crop the ECG region (full width, specific height range)
    crop_box = (0, 400, img_width, 1500)  # Adjust height range as needed

    # Crop and save the ECG image
    ecg_image = last_page.crop(crop_box)
    ecg_image.save(save_ecg_image, "PNG")
    print(f"ECG image saved at: {save_ecg_image}")

    return save_ecg_image


if __name__ == "__main__":
    pass
