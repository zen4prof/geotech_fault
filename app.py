
import streamlit as st
from PIL import Image as PILImage, ImageDraw
import os
import numpy as np
import onnxruntime

# Set up the Streamlit application title
st.title("Geotechnical Fault Detection Web App (ONNX)")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Path to the exported ONNX model - assuming it's in the same directory as the app.py for deployment
ONNX_MODEL_PATH = 'best.onnx' # This path needs to be relative to where app.py is deployed

@st.cache_resource # Cache the model loading
def load_onnx_model(path):
    if os.path.exists(path):
        try:
            # Load the ONNX model using onnxruntime
            session = onnxruntime.InferenceSession(path, None)
            # Get input and output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            return session, input_name, output_name
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
            return None, None, None
    else:
        st.error(f"ONNX model file not found at {path}")
        return None, None, None

session, input_name, output_name = load_onnx_model(ONNX_MODEL_PATH)

# Class names - Get from data.yaml or training results
CLASS_NAMES = ['Block loss', 'Crack on Ashpat', 'Long Crack', 'Opening on the wall', 'Vegetation on Wall', 'Vegetation on slope', 'Vertical Crack', 'Wall deformation', 'bad foundation', 'corrosion', 'slope deformation']

# Display the uploaded image and run inference
if uploaded_file is not None:
    image = PILImage.open(uploaded_file).convert('RGB') # Ensure image is RGB
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if session:
        st.write("Running inference using ONNX model...")
        # Preprocess the image for ONNX model input
        img = np.array(image)
        # Resize image to model input size (assuming 640x640)
        img_resized = PILImage.fromarray(img).resize((640, 640))
        img_resized = np.array(img_resized)

        img_processed = img_resized[:, :, ::-1].transpose(2, 0, 1)  # Convert HWC to CHW, RGB to BGR
        img_processed = np.ascontiguousarray(img_processed)
        img_processed = img_processed.astype(np.float32)
        img_processed /= 255.0  # Normalize to [0, 1]
        img_processed = np.expand_dims(img_processed, 0) # Add batch dimension

        try:
            # Run inference
            onnx_inputs = {input_name: img_processed}
            onnx_outputs = session.run([output_name], onnx_inputs)
            predictions = onnx_outputs[0] # Get the output tensor

            # Postprocess the results
            # YOLOv8 ONNX output format is typically (batch_size, num_boxes, 5 + num_classes)
            # or (batch_size, 5 + num_classes, num_boxes) - need to check the exact shape
            # Based on previous output (1, 15, 8400), it seems to be (batch_size, 5 + num_classes, num_boxes)
            # Transpose to (batch_size, num_boxes, 5 + num_classes)
            predictions = predictions.transpose(0, 2, 1) # Shape becomes (1, 8400, 15)
            predictions = predictions[0] # Remove batch dimension, shape is (8400, 15)


            confidence_threshold = 0.25
            # NMS is typically a separate step after getting raw predictions
            # For simplicity in visualization without a full NMS implementation here,
            # we'll just filter by confidence and draw boxes.
            # A proper deployment would include an NMS step.

            boxes = predictions[:, :4]
            confidences = np.max(predictions[:, 4:], axis=1) # Max class score as confidence for this simplified approach
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            # Apply confidence threshold
            valid_detections = confidences > confidence_threshold

            boxes = boxes[valid_detections]
            class_ids = class_ids[valid_detections]
            confidences = confidences[valid_detections]

            # Convert box format from [cx, cy, w, h] to [x1, y1, x2, y2]
            # Scale boxes to original image size
            original_width, original_height = image.size
            img_size_model = 640 # Assuming model input size is 640x640

            boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * (original_width / img_size_model) # x1
            boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * (original_height / img_size_model) # y1
            boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) * (original_width / img_size_model)     # x2
            boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) * (original_height / img_size_model)     # y2


            # Draw bounding boxes on the original image
            draw = ImageDraw.Draw(image)
            colors = {} # Dictionary to store colors for each class

            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(class_ids[i])
                confidence = confidences[i]
                label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"

                # Assign a color to the class if not already assigned
                if class_id not in colors:
                     # Generate a random color for each class
                     colors[class_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                color = colors[class_id]

                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=2)
                draw.text((box[0], box[1]), label, fill=color)


            st.write("Inference Results:")
            st.image(image, caption="Image with Detections.", use_column_width=True)


        except Exception as e:
            st.error(f"Error during ONNX inference or postprocessing: {e}")

    else:
        st.warning("ONNX Model could not be loaded, cannot perform inference.")


# Add a footer or additional information
st.write("---")
st.write("Powered by YOLOv8 (ONNX)")
