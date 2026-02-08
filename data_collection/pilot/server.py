import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
CONN_STR = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "pilot-data"

if not CONN_STR:
    print("WARNING: AZURE_STORAGE_CONNECTION_STRING not found in .env file")

@app.route('/')
def index():
    # Serve the app.html file when accessing root
    return send_from_directory('.', 'app.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        data = request.json
        rows = data.get('logs', [])
        
        if not rows:
            return jsonify({"status": "error", "message": "No data to save"}), 400

        # Generate CSV Content
        csv_content = "Label,StartTimestamp_Unix_Ms,EndTimestamp_Unix_Ms\n"
        for row in rows:
            # Ensure integers
            start_ts = int(round(row['start']))
            end_ts = int(round(row['end']))
            csv_content += f"{row['label']},{start_ts},{end_ts}\n"

        # Unique filename
        filename = f"activity_session_{uuid.uuid4()}.csv"

        # Upload to Azure
        blob_service_client = BlobServiceClient.from_connection_string(CONN_STR)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()

        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(csv_content)

        print(f"Uploaded {filename} with {len(rows)} rows.")
        return jsonify({"status": "success", "filename": filename})

    except Exception as e:
        print(f"Error uploading: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from iPhone on the same network
    print("Starting server at http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
