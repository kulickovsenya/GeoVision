import pypdfium2 as pdfium
import os
from PIL import Image
import cv2


def convert_pdf_to_jpeg(filename: str) -> str:
    pdf = pdfium.PdfDocument(filename)
    # Loop over pages and render
    for i in range(len(pdf)):
        page = pdf[i]
        image = page.render(scale=4).to_pil()
        path = filename.split('\\')[:-1]
        name = filename.split('\\')[-1]
        # filename_out = os.path.join(*path, "temp_files", f"{path[-1]}_{i:03d}.jpg")
        # if not os.path.isdir(os.path.join(*path, "temp_files")):
        #     os.mkdir(os.path.join(*path, "temp_files"))

        filename_out = os.path.join("temp_files", f"{name[:-4]}_{i:03d}.jpg")
        if not os.path.isdir(os.path.join("temp_files")):
            os.mkdir(os.path.join("temp_files"))
        image.save(filename_out)
        return filename_out


def convert_tiff_to_jpeg(filename: str) -> str:
    image = Image.open(filename)
    name = filename.split('\\')[-1]
    filename_out = os.path.join("temp_files", f"{name[:-4]}_{i:03d}.jpg")
    if not os.path.isdir(os.path.join("temp_files")):
        os.mkdir(os.path.join("temp_files"))
    image.save(filename_out)
    return filename_out


def convert_png_to_jpeg(filename: str) -> str:
    image = cv2.cvtColor(filename, cv2.COLOR_RGBA2BGR)
    name = filename.split('\\')[-1]
    filename_out = os.path.join("temp_files", f"{name[:-4]}_{i:03d}.jpg")
    if not os.path.isdir(os.path.join("temp_files")):
        os.mkdir(os.path.join("temp_files"))
    image.save(filename_out)
    return filename_out
