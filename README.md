# Font Identifier API

A Flask API that identifies fonts in images using Roboflow.

## Setup

1. Create a virtual environment: 
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate

2. Install dependencies:
pip install -r requirements.txt

3. Create a `.env` file with your Roboflow credentials:
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE_ID=your_workspace_id
ROBOFLOW_PROJECT_ID=your_project_id
ROBOFLOW_MODEL_VERSION=1

4. Run the app:
python app.py

The API will be available at http://localhost:5000.