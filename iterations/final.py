# import cv2
# import pytesseract
# import numpy as np
# import matplotlib.pyplot as plt

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# def preprocess_and_extract_readings(image_path, bounding_boxes):
#     # Read the image
#     img = cv2.imread(image_path)
    
#     # Resize the image to a standard size
#     resized = cv2.resize(img, (224, 224))
    
#     # Get the dimensions of the resized image
#     height, width, _ = resized.shape
    
#     # Define the cropping area (top center)
#     roi_height = int(height * 0.7)  # Height of the ROI
#     roi_width = int(width * 0.7)    # Width of the ROI
#     x_start = int((width - roi_width) / 3)
#     y_start = 0  # Starting from the top
    
#     # Crop the region of interest
#     cropped = resized[y_start:y_start + roi_height, x_start:x_start + roi_width]
    
#     # Draw the bounding boxes on the cropped image
#     for bbox in bounding_boxes:
#         digit, x1, y1, x2, y2, x3, y3, x4, y4 = bbox
#         # Convert normalized coordinates to absolute coordinates
#         x1 = int(x1 * roi_width)
#         y1 = int(y1 * roi_height)
#         x2 = int(x2 * roi_width)
#         y2 = int(y2 * roi_height)
#         x3 = int(x3 * roi_width)
#         y3 = int(y3 * roi_height)
#         x4 = int(x4 * roi_width)
#         y4 = int(y4 * roi_height)
        
#         # Draw the bounding box
#         pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
#         pts = pts.reshape((-1, 2, 2))
#         cv2.polylines(cropped, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
#     # Convert the cropped image to grayscale
#     gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
#     # Use PyTesseract to extract text from the cropped image
#     custom_config = r'--oem 3 --psm 6 outputbase digits'
#     extracted_text = pytesseract.image_to_string(gray, config=custom_config)
    
#     # Print the extracted numerical values
#     print("Extracted Readings:", extracted_text)
    
#     # Display the original image, cropped image, and image with bounding boxes
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title('Original Image')
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.subplot(1, 3, 2)
#     plt.title('Cropped Image')
#     plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.subplot(1, 3, 3)
#     plt.title('Bounding Boxes')
#     plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.show()

# # Example usage
# image_path = 'data_emr\Mtr2_jpg.rf.3f206996746fed5509a956966986f26c.jpg'

# # Bounding box data for the readings
# bounding_boxes = [
#     [2, 0.33281250000000007, 0.2453125, 0.28125, 0.2453125, 0.28125000000000006, 0.4984375, 0.33281250000000007, 0.4984375],
#     [1, 0.38750000000000007, 0.24375000000000005, 0.3265625, 0.24375, 0.32656250000000003, 0.4953125, 0.38750000000000007, 0.4953125],
#     [6, 0.446875, 0.24375000000000008, 0.384375, 0.24375000000000008, 0.384375, 0.4859375, 0.446875, 0.4859375],
#     [5, 0.509375, 0.2406250000000001, 0.4453125, 0.2406250000000001, 0.4453125, 0.4921875, 0.509375, 0.4921875],
#     [8, 0.5640625, 0.24375000000000008, 0.5140625, 0.24375000000000008, 0.5140625, 0.4890625, 0.5640625, 0.4890625],
#     [8, 0.6265625, 0.24218750000000008, 0.5734375, 0.24218750000000008, 0.5734375, 0.484375, 0.6265625, 0.484375],
#     [1, 0.6859375, 0.2390625000000001, 0.6234375, 0.2390625000000001, 0.6234375, 0.49375, 0.6859375, 0.49375],
#     [0, 0.584375, 0.4328125, 0.546875, 0.4328125, 0.546875, 0.496875, 0.584375, 0.496875]
# ]

# preprocess_and_extract_readings(image_path, bounding_boxes)

# import cv2
# import pytesseract
# import numpy as np
# import matplotlib.pyplot as plt

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# def preprocess_and_extract_readings(image_path, bounding_boxes):
#     # Read the image
#     img = cv2.imread(image_path)
    
#     # Resize the image to a standard size
#     resized = cv2.resize(img, (224, 224))
    
#     # Get the dimensions of the resized image
#     height, width, _ = resized.shape
    
#     # Define the cropping area (center)
#     roi_height = 250  # Height of the ROI
#     roi_width = 200   # Width of the ROI
#     x_start = 87 - roi_width // 2
#     y_start = 39 - roi_height // 2
    
#       # Ensure the cropping area is within the image boundaries
#     x_start = max(0, x_start)
#     y_start = max(0, y_start)
#     x_end = min(width, x_start + roi_width)
#     y_end = min(height, y_start + roi_height)
    
#     # Crop the region of interest
#     cropped = resized[y_start:y_end, x_start:x_end]    
#     # Draw the bounding boxes on the cropped image
#     for bbox in bounding_boxes:
#         digit, x1, y1, x2, y2, x3, y3, x4, y4 = bbox
#         # Convert normalized coordinates to absolute coordinates
#         x1 = int(x1 * roi_width) + 2
#         y1 = int(y1 * roi_height) + y_start
#         x2 = int(x2 * roi_width) + 2
#         y2 = int(y2 * roi_height) + y_start
#         x3 = int(x3 * roi_width) + 2
#         y3 = int(y3 * roi_height) + y_start
#         x4 = int(x4 * roi_width) + 2
#         y4 = int(y4 * roi_height) + y_start 
        
#         # Draw the bounding box
#         pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
#         pts = pts.reshape((-1, 2))
#         cv2.polylines(cropped, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    
#     # Convert the cropped image to grayscale
#     gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
#     # Use PyTesseract to extract text from the cropped image
#     custom_config = r'--oem 3 --psm 6 outputbase digits'
#     extracted_text = pytesseract.image_to_string(gray, config=custom_config)
    
#     # Print the extracted numerical values
#     print("Extracted Readings:", extracted_text)
    
#     # Display the original image, cropped image, and image with bounding boxes
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title('Original Image')
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.subplot(1, 3, 2)
#     plt.title('Cropped Image')
#     plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.subplot(1, 3, 3)
#     plt.title('Bounding Boxes')
#     plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
    
#     plt.show()

# # Example usage
# image_path = 'data_emr\Mtr2_jpg.rf.3f206996746fed5509a956966986f26c.jpg'

# # Bounding box data for the readings
# bounding_boxes = [
#     [2, 0.33281250000000007, 0.2453125, 0.28125, 0.2453125, 0.28125000000000006, 0.4984375, 0.33281250000000007, 0.4984375],
#     [1, 0.38750000000000007, 0.24375000000000005, 0.3265625, 0.24375, 0.32656250000000003, 0.4953125, 0.38750000000000007, 0.4953125],
#     [6, 0.446875, 0.24375000000000008, 0.384375, 0.24375000000000008, 0.384375, 0.4859375, 0.446875, 0.4859375],
#     [5, 0.509375, 0.2406250000000001, 0.4453125, 0.2406250000000001, 0.4453125, 0.4921875, 0.509375, 0.4921875],
#     [8, 0.5640625, 0.24375000000000008, 0.5140625, 0.24375000000000008, 0.5140625, 0.4890625, 0.5640625, 0.4890625],
#     [8, 0.6265625, 0.24218750000000008, 0.5734375, 0.24218750000000008, 0.5734375, 0.484375, 0.6265625, 0.484375],
#     [1, 0.6859375, 0.2390625000000001, 0.6234375, 0.2390625000000001, 0.6234375, 0.49375, 0.6859375, 0.49375],
#     [0, 0.584375, 0.4328125, 0.546875, 0.4328125, 0.546875, 0.496875, 0.584375, 0.496875]
# ]

# preprocess_and_extract_readings(image_path, bounding_boxes)

import cv2
import pytesseract
import numpy as np

def preprocess_and_extract_readings(image_path, bounding_boxes):
    # Read the image
    input_image = cv2.imread(image_path)

    # Convert BGR to grayscale:
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Threshold via Otsu:
    thresh_value, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours:
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the input image to draw bounding boxes
    output_image = input_image.copy()

    # Look for the outer bounding boxes (no children):
    for _, c in enumerate(contours):
        # Get the bounding rectangle of the current contour:
        bound_rect = cv2.boundingRect(c)

        # Get the bounding rectangle data:
        rect_x = bound_rect[0]
        rect_y = bound_rect[1]
        rect_width = bound_rect[2]
        rect_height = bound_rect[3]

        # Estimate the bounding rect area:
        rect_area = rect_width * rect_height

        # Set a min area threshold
        min_area = 10

        # Filter blobs by area:
        if rect_area > min_area:
            # Draw bounding box:
            color = (0, 255, 0)
            cv2.rectangle(output_image, (int(rect_x), int(rect_y)),
                          (int(rect_x + rect_width), int(rect_y + rect_height)), color, 2)

            # Crop bounding box:
            current_crop = input_image[rect_y:rect_y+rect_height,rect_x:rect_x+rect_width]

            # Use PyTesseract for OCR:
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            extracted_text = pytesseract.image_to_string(current_crop, config=custom_config)
            print("Extracted Text:", extracted_text)

    # Display the output:
    cv2.imshow("Bounding Boxes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Set image path
image_path = 'data_emr\Mtr2_jpg.rf.3f206996746fed5509a956966986f26c.jpg'

# Call the function
preprocess_and_extract_readings(image_path, None)