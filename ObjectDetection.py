import os
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import random


def detect_objects(image_path):
    """Detect objects in an image using a pre-trained YOLO model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        results: Results from the object detection model.
        
    """
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        
    # Load a pre-trained YOLO model (using YOLOv8 in this case)
    model = YOLO("yolov8m.pt")  # Using the medium version for this example
    
    # Perform object detection on the image
    results = model(image_path)
    
    """Show the image with detected objects and confidence rates.

    Args:
        results: Results from the object detection model.
        image_path (str): Path to the original image file.
    """
    # Display the image with bounding boxes and labels
    fig, ax = plt.subplots()

    # Open the image file
    with Image.open(image_path) as img:
        ax.imshow(img)

    # Draw bounding boxes, labels, and confidence scores
    for result in results:
        for box in result.boxes:
            # Ensure proper conversion of tensors to standard Python types
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list
            label = int(box.cls)  # Convert to integer
            conf = float(box.conf)  # Convert to float
            label_text = f"{result.names[label]}: {conf:.2f}"  # Proper string formatting
            
            # Change color based on confidence:
            if conf > 0.5:
                ax.text(x1, y1, label_text, color='white', bbox=dict(facecolor='green', alpha=0.5))
            elif conf == 0.5:
                ax.text(x1, y1, label_text, color='white', bbox=dict(facecolor='yellow', alpha=0.5))
            else:
                ax.text(x1, y1, label_text, color='white', bbox=dict(facecolor='red', alpha=0.5))

            # Creating an array of color for edge of result boxes:
            color_arr = ['r','g','b','y']
            
            # Draw the bounding box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_arr[random.randint(0,3)], facecolor='none')
            ax.add_patch(rect)

            # Add the label with confidence score
            

    plt.show()
