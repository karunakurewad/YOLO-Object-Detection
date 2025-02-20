import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load YOLO model and classes
def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    with open(names_path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return net, classes

# Detect objects using YOLO
def detect_objects(net, image, classes):
    inpWidth = 416
    inpHeight = 416
    blob = cv2.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    yolo_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(yolo_layers)

    boxes, confidences, class_ids = [], [], []
    frameHeight, frameWidth = image.shape[:2]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:  # Confidence threshold
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.2)  # Non-max suppression
    return boxes, confidences, class_ids, indexes

# Draw bounding boxes on the image
def draw_boxes(image, boxes, confidences, class_ids, indexes, classes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confi = str(round(confidences[i], 2))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
            cv2.putText(image, f"{label} {confi}", (x, y - 10), font, 1, (255, 255, 255), 2)
    return image

# Streamlit app
def main():
    st.title("YOLO Object Detection")

    # Paths to YOLO model files
    cfg_path = "yolov3-tiny.cfg"
    weights_path = "yolov3-tiny.weights"
    names_path = "coco.names"

    # Load YOLO model
    net, classes = load_yolo_model(cfg_path, weights_path, names_path)

    # File upload for image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert uploaded image to an array
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Perform object detection
        boxes, confidences, class_ids, indexes = detect_objects(net, image, classes)

        # Draw bounding boxes on the image
        detected_image = draw_boxes(image.copy(), boxes, confidences, class_ids, indexes, classes)

        # Display images: original and detected
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, caption="Original Image", use_column_width=True)

        with col2:
            st.subheader("Detected Objects")
            st.image(detected_image, caption="Detected Image", use_column_width=True)

if __name__ == "__main__":
    main()
