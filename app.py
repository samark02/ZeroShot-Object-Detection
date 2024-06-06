import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import random
import tempfile

# Initialize YOLO model with the specified weights file
model = YOLO("yolov8x-worldv2.pt")

# Streamlit App title
st.title("Zero-Shot Object Detection using Yolo-World")

# User input for classes, separated by commas
custom_classes = st.text_input("Enter classes (comma-separated)", "person, backpack, ball, cat, building, football")
# Split the input string into a list of classes and remove any extra whitespace
classes = [cls.strip() for cls in custom_classes.split(",")]
# Set the classes for the YOLO model
model.set_classes(classes)

# Function to generate distinct colors for each class
def generate_colors(num_colors):
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_colors)]

# Generate a unique color for each class and map them
class_colors = {cls: color for cls, color in zip(classes, generate_colors(len(classes)))}

# Function to annotate image with detections
def annotate_image(image, results):
    for box in results[0].boxes:  # Extract bounding boxes from the results
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = box.conf[0]  # Confidence score of the detection
        cls_id = int(box.cls[0])  # Class ID of the detected object
        label = f"{results[0].names[cls_id]} {conf:.2f}"  # Label with class name and confidence
        color = class_colors[results[0].names[cls_id]]  # Color for the class

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Calculate the text size and position
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)  # Background rectangle for label
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Text label in black color

    return image

# Image upload and processing
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)  # Open the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)  # Display the uploaded image

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run inference on the image
    results = model(image_cv)

    # Annotate the image with detections
    annotated_image = annotate_image(image_cv, results)

    # Convert annotated image back to RGB for displaying in Streamlit
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Objects.", use_column_width=True)

    # Annoted Image Download Option
    # Convert annotated image to PIL format
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    # Create a download button for the annotated image
    st.download_button(
        label="Download Annotated Image",
        data=cv2.imencode('.jpg', cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="annotated_image.jpg",
        mime="image/jpeg"
    )


# Video upload and processing
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_video:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Open the video file
    video_cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()  # Placeholder for displaying video frames

    # Initialize VideoWriter for saving the output video
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while video_cap.isOpened():
        ret, frame = video_cap.read()  # Read a frame from the video
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Annotate the frame with detections
        annotated_frame = annotate_image(frame, results)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Convert annotated frame back to RGB for displaying in Streamlit
        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    video_cap.release()  # Release the video capture object
    out.release()  # Release the VideoWriter object

    # Create a download button for the annotated video
    with open(output_video_path, "rb") as video_file:
        st.download_button(
            label="Download Annotated Video",
            data=video_file,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )
