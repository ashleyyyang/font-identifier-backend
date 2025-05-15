from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from dotenv import load_dotenv
import logging
from roboflow import Roboflow
import cv2
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Font database
FONT_LINKS = {
    "inter": "https://fonts.google.com/specimen/Inter",
    "inter bold": "https://fonts.google.com/specimen/Inter",
    "instrument serif regular": "https://fonts.google.com/specimen/Instrument+Serif",
    "instrument serif regular italic": "https://fonts.google.com/specimen/Instrument+Serif",
}

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    logger.info("Root endpoint accessed")
    return jsonify({"message": "Font Identifier API is running. Use /api/test or /api/identify endpoints."})

# Test endpoint
@app.route('/api/test', methods=['GET'])
def test():
    logger.info("Test endpoint accessed")
    return jsonify({"message": "Python backend is working!"})

# Font identification endpoint
@app.route('/api/identify', methods=['POST'])
def identify_font():
    logger.info("Identify endpoint accessed")
    
    if 'image' not in request.files:
        logger.warning("No image in request")
        return jsonify({"success": False, "error": "No image uploaded"})
    
    file = request.files['image']
    
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({"success": False, "error": "Empty file"})
    
    temp_file = None
    preprocessed_path = None
    
    try:
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_file.name)
        temp_file.close()
        
        logger.info(f"Saved image to temporary file: {temp_file.name}")
        
        # Preprocess the image - resize to 640x640 with padding
        # Read the image
        image = cv2.imread(temp_file.name)
        h, w = image.shape[:2]
        
        # Create square canvas with padding (640x640)
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        scale = min(640 / w, 640 / h) * 0.9  # 0.9 to leave some padding
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Center the image
        x_offset = (640 - new_w) // 2
        y_offset = (640 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Save the preprocessed image
        preprocessed_path = temp_file.name + "_preprocessed.jpg"
        cv2.imwrite(preprocessed_path, canvas)
        
        logger.info(f"Preprocessed image saved to: {preprocessed_path}")
        
        # Initialize Roboflow
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        workspace_id = os.environ.get("ROBOFLOW_WORKSPACE_ID")
        project_id = os.environ.get("ROBOFLOW_PROJECT_ID")
        model_version = os.environ.get("ROBOFLOW_MODEL_VERSION", "1")
        
        logger.info(f"Initializing Roboflow with Project: {project_id}, Version: {model_version}")
        
        if not all([api_key, workspace_id, project_id]):
            logger.warning("Roboflow credentials not found, using mock response")
            # Use mock response if credentials are missing
            import random
            fonts = ["Inter", "Inter Bold", "Instrument Serif Regular", "Verdana"]
            font = random.choice(fonts)
            confidence = random.uniform(75.0, 95.0)
            
            # Clean up the temporary files
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            if preprocessed_path and os.path.exists(preprocessed_path):
                os.unlink(preprocessed_path)
            
            return jsonify({
                "success": True,
                "font": font,
                "confidence": confidence,
                "font_link": FONT_LINKS.get(font.lower(), "https://fonts.google.com/")
            })
        
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_id).project(project_id)
        model = project.version(model_version).model
        
        # Get prediction from Roboflow
        logger.info("Sending preprocessed image to Roboflow for prediction")
        prediction = model.predict(preprocessed_path, confidence=40, overlap=30).json()
        logger.info(f"Received prediction from Roboflow: {prediction}")
        
        # Clean up the temporary files
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.unlink(preprocessed_path)
        
        # Process the prediction
        if "predictions" in prediction and len(prediction["predictions"]) > 0:
            # Sort predictions by confidence
            sorted_predictions = sorted(
                prediction["predictions"], 
                key=lambda x: x.get("confidence", 0),
                reverse=True
            )
            
            # Get top prediction
            top_prediction = sorted_predictions[0]
            font_name = top_prediction.get("class", "Unknown")
            confidence = top_prediction.get("confidence", 0) * 100  # Convert to percentage
            
            logger.info(f"Top prediction: {font_name} with {confidence:.2f}% confidence")
            
            # Get font link
            font_link = FONT_LINKS.get(font_name.lower(), "https://fonts.google.com/")
            
            return jsonify({
                "success": True,
                "font": font_name,
                "confidence": confidence,
                "font_link": font_link
            })
        else:
            logger.warning("No predictions found in Roboflow response")
            return jsonify({
                "success": False,
                "error": "No fonts detected in the image"
            })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        
        # Attempt to clean up the temporary files
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")
                
        if preprocessed_path and os.path.exists(preprocessed_path):
            try:
                os.unlink(preprocessed_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up preprocessed file: {str(cleanup_error)}")
        
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Main entry point
# At the bottom of your app.py
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production