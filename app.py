import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import os


# Set up the model configuration
def setup_model():
    cfg = get_cfg()

    # Load the pre-trained Faster R-CNN model configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

    # Check if the model file exists
    model_path = "output_mri/model_final.pth"
    if not os.path.exists(model_path):
        st.error(f"Model weights not found at {model_path}. Please check the path.")
        return None

    cfg.MODEL.WEIGHTS = model_path  # Use the path where your trained model is stored
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Update this to match your dataset classes

    return DefaultPredictor(cfg)


# Initialize the model
predictor = setup_model()

if predictor is not None:  # Only proceed if the model is correctly set up
    # Streamlit app interface
    st.title("Brain MRI Tumor Detection")
    st.write("Upload an MRI image to detect if it shows a tumor, negative, or positive findings.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert uploaded image to OpenCV format
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Convert the image to BGR format if necessary
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Run prediction
            outputs = predictor(image_np)

            # Ensure metadata is registered, otherwise handle gracefully
            dataset_name = "mri_test"  # Make sure this matches your dataset name
            try:
                metadata = MetadataCatalog.get(dataset_name)
            except KeyError:
                st.error(f"Dataset '{dataset_name}' not found. Please ensure it's registered.")
                metadata = None

            # Visualize the prediction if metadata is available
            if metadata:
                v = Visualizer(image_np[:, :, ::-1],  # Convert BGR to RGB for visualization
                               metadata=metadata,
                               scale=0.8,
                               instance_mode=ColorMode.IMAGE_BW)  # Greyscale unsegmented parts
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                # Convert visualization to PIL image
                predicted_image = Image.fromarray(v.get_image()[:, :, ::-1])  # Convert RGB back to BGR

                # Display original and predicted images side by side
                st.image([image, predicted_image], caption=['Original Image', 'Predicted Image'], use_column_width=True)

                # Get the prediction results
                #instances = outputs["instances"]
                #pred_classes = instances.pred_classes.tolist()

                # Define the class labels
                #class_labels = {2: "positive"}

                # Show the prediction results
                #st.write("### Prediction Results")
                #for pred_class in pred_classes:
                #   st.write(f"Detected: **{class_labels[pred_class]}**")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Provide clear explanations of the class labels
    st.write("### Explanation of Class Labels:")
    st.write("- **Positive**: Indicates the presence of an abnormality, but it's not necessarily a tumor.")
