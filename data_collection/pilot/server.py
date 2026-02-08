import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
DATA_DIR = "/mnt/pilot-app"  # Docker volume mount point

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

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

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"activity_session_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        
        # Save to local filesystem
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(csv_content)

        print(f"Saved {filename} with {len(rows)} rows to {filepath}")
        return jsonify({"status": "success", "filename": filename})

    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from external devices
    print(f"Starting server at http://0.0.0.0:8003")
    print(f"Data will be saved to: {DATA_DIR}")
    app.run(host='0.0.0.0', port=8003, debug=True)
