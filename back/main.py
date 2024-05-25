import os.path

from tools import convert_pdf_to_jpeg, convert_tiff_to_jpeg, convert_png_to_jpeg

import cv2
from scipy.ndimage import rotate as rotate_image

import numpy as np
import matplotlib.pyplot as plt


def conversion_to_jpeg(filename: str) -> str:
    if filename[-3:].lower() == 'pdf':
        filename_out = convert_pdf_to_jpeg(filename)
    if filename[-4:].lower() == 'tiff':
        filename_out = convert_tiff_to_jpeg(filename)
    if filename[-3:].lower() == 'png':
        filename_out = convert_png_to_jpeg(filename)
    # TODO: добавить обработку CDR файлов
    return filename_out


def find_intersections(lines, img_shape):
    """ Finds the intersections of horizontal and vertical lines. """

    def line_intersection(line1, line2):
        """ Finds the intersection of two lines given in (rho, theta) format. """
        rho1, theta1 = line1
        rho2, theta2 = line2
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        A = np.array([[a1, b1], [a2, b2]])
        b = np.array([rho1, rho2])
        try:
            x0, y0 = np.linalg.solve(A, b)
            return int(np.round(x0)), int(np.round(y0))
        except np.linalg.LinAlgError:
            return None

    horizontal_lines = [line for line in lines if abs(np.sin(line[0][1])) > 0.5]
    vertical_lines = [line for line in lines if abs(np.cos(line[0][1])) > 0.5]

    intersections = []
    for hline in horizontal_lines:
        for vline in vertical_lines:
            point = line_intersection(hline[0], vline[0])
            if point is not None and 0 <= point[0] < img_shape[1] and 0 <= point[1] < img_shape[0]:
                intersections.append(point)
    return intersections

def crop_image_by_segmentation(image):
    # Load the image
    # image = cv2.imread('your_image.jpg')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Find intersections of the lines
    intersections = find_intersections(lines, image.shape)

    # Sort intersections by their coordinates
    intersections = sorted(intersections, key=lambda x: (x[1], x[0]))

    # Assuming intersections are found, and there are enough to form regions
    if len(intersections) < 4:
        print("Not enough intersections found to form zones.")
    else:
        # Crop and save each zone
        for i in range(len(intersections) - 1):
            for j in range(len(intersections) - 1):
                top_left = intersections[i]
                bottom_right = intersections[i + 1]
                crop_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                cv2.imwrite(f'crop_{i}_{j}.jpg', crop_img)

    # Optionally, display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extraire_different_colors(img, color):

    # Split the image into its BGR components
    blue, green, red = cv2.split(img)

    # Create an empty image with the same shape as the original
    zeros = np.zeros_like(blue)

    # Merge the channels, but keep only the red channel active
    red_only_image = cv2.merge([zeros, zeros, red])
    green_only_image = cv2.merge([zeros, green, zeros])
    blue_only_image = cv2.merge([blue, zeros, zeros])

    # cv2.imshow('Red Only Image', red_only_image)
    # cv2.imshow('Green Only Image', green_only_image)
    # cv2.imshow('Blue Only Image', blue_only_image)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the range for black color (0-50 for each channel)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])

    # Create a mask that identifies the black regions in the image
    black_mask = cv2.inRange(img, lower_black, upper_black)

    # Create an empty image with the same shape as the original
    # Create a white image with the same shape as the original
    white_image = np.ones_like(img) * 255

    # Apply the mask to the white image, setting the black regions
    white_image[black_mask == 255] = [0, 0, 0]

    # # Highlight the black regions by keeping them as they are and setting other regions to white
    # highlight_black[black_mask == 255] = img[black_mask == 255]
    #
    # # Save the resulting image
    # cv2.imwrite('highlight_black_image.jpg', highlight_black)

    # Optionally, display the images
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Black Highlighted Image', white_image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
    if color =='red':
        return red_only_image
    elif color =='blue':
        return blue_only_image
    elif color == 'green':
        return green_only_image
    elif color == 'grey':
        return gray_image
    elif color == 'black':
        return white_image



def detect_long_horizontal_line(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    # print(edges)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # print(lines)

    # Initialize variables to hold the longest horizontal line's coordinates and length
    longest_line = None
    max_length = 0

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if abs(np.sin(theta)) > 0.5:  # Check if the line is more horizontal than vertical
                # Calculate the endpoints of the line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Calculate the length of the line
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Update the longest line if the current one is longer
                if line_length > max_length:
                    max_length = line_length
                    longest_line = ((x1, y1), (x2, y2))


    if longest_line is not None:
        print(f"The longest horizontal line coordinates are: {longest_line}")
        # Optionally, draw the longest horizontal line on the image
        cv2.line(image, longest_line[0], longest_line[1], (0, 0, 255), 2)
        # cv2.imshow('Longest Horizontal Line', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return x1, y1, x2, y2
    else:
        print("No horizontal line found")
        return 0


def crop_image(image, x1, y1, x2, y2):
    # Ensure the coordinates are within the image dimensions
    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        raise ValueError("The specified coordinates are out of image bounds")

    # Crop the image using array slicing
    cropped_image = image[y1:y2, x1:x2]

    # Save the cropped image
    # cv2.imwrite(output_path, cropped_image)

    return cropped_image




if __name__ == '__main__':
    filename = r"GeoVision_dataset\well_1.PDF"
    filename = conversion_to_jpeg(filename)
    print(filename)

    img = cv2.imread(filename)
    if img is None:
        raise ValueError("Image not found or path is incorrect")
    image_hor = rotate_image(img, 90)

    x1, y1, x2, y2 = detect_long_horizontal_line(image_hor)
    # Example usage
    img = crop_image(img, x1, y1, x2, y2)

    extraire_different_colors(image_hor)

    #
    # # Display the original and the line-detected image
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Original Image')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

