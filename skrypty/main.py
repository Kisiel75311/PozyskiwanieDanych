import cv2
import os

# Directory for output
output_directory = "tmp2"

# Create directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Range of values to be tested
block_sizes = list(range(3, 150, 2))  # Odd values from 3 to 49
Cs = list(range(-10, 10, 2))  # Values from -10 to 10 in steps of 2

# Load the image (unchanged)
image_path = '2.jpeg'  # Update this path if needed
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
yellow = [0, 255, 255]  # Color for highlighting

for block_size in block_sizes:
    for C in Cs:
        # Apply adaptive binarization with current combination of values
        adaptive_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)

        # Invert the binary image
        inverted_binary = cv2.bitwise_not(adaptive_binary)

        # Apply morphological operations
        cleaned_adaptive = cv2.morphologyEx(inverted_binary, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned_adaptive = cv2.morphologyEx(cleaned_adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Highlight isolated lung areas with yellow on a copy of the original image
        highlighted_adaptive = image.copy()
        highlighted_adaptive[cleaned_adaptive == 255] = yellow

        # Save the resulting image to the output directory
        output_path = os.path.join(output_directory, f"highlighted_block_{block_size}_C_{C}.png")
        cv2.imwrite(output_path, highlighted_adaptive)

print("Finished generating inverse binarization images for different combinations of values.")
