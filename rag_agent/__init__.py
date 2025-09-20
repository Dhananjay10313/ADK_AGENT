"""
Vertex AI RAG Agent

A package for interacting with Google Cloud Vertex AI RAG capabilities.
"""

import os

import vertexai
from dotenv import load_dotenv
import json
import tempfile
import atexit
import os

# Load environment variables
load_dotenv()

# Get Vertex AI configuration from environment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

# Initialize Vertex AI at package load time
try:
    # Support service account JSON provided via env var or file path.
    # Priority:
    # 1. If GOOGLE_APPLICATION_CREDENTIALS is set and points to a file, leave it.
    # 2. If GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON) is provided, write to a temp file and set GOOGLE_APPLICATION_CREDENTIALS.
    # 3. If GOOGLE_APPLICATION_CREDENTIALS_JSON (alternate name) is provided, treat like raw JSON.
    sa_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if sa_json and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        # If it looks like a path to a file, use it directly
        if os.path.exists(sa_json):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_json
        else:
            # Otherwise treat content as JSON string and write to temp file
            try:
                parsed = json.loads(sa_json)
                fd, tmp_path = tempfile.mkstemp(prefix='gcp-sa-', suffix='.json')
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = tmp_path
                # ensure temp file is removed on process exit
                atexit.register(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))
            except Exception:
                # if it's not JSON, skip and warn via print below
                pass

    if PROJECT_ID and LOCATION:
        print(f"Initializing Vertex AI with project={PROJECT_ID}, location={LOCATION}")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print("Vertex AI initialization successful")
    else:
        print(
            f"Missing Vertex AI configuration. PROJECT_ID={PROJECT_ID}, LOCATION={LOCATION}. "
            f"Tools requiring Vertex AI may not work properly."
        )
except Exception as e:
    print(f"Failed to initialize Vertex AI: {str(e)}")
    print("Please check your Google Cloud credentials and project settings.")

# Import agent after initialization is complete
from . import agent