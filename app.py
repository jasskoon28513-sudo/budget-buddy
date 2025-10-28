import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google.generativeai.errors import APIError

# --- Configuration and Initialization ---

# Azure App Service will securely provide the API key via Application Settings.
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Hardcoded model name as requested
MODEL_TO_USE = 'gemini-2.5-flash' 

if not API_KEY:
    # Print a warning for the Azure logs if the key is missing
    print("FATAL: GOOGLE_API_KEY environment variable not found. The application cannot start.")

try:
    # Configure and initialize the model only if the API key is present
    if API_KEY:
        genai.configure(api_key=API_KEY)
        # The GenerativeModel instance explicitly uses gemini-2.5-flash
        model = genai.GenerativeModel(MODEL_TO_USE)
    else:
        # Placeholder if configuration failed
        model = None 
except Exception as e:
    # Log configuration errors
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    model = None

# Initialize Flask app
# The name must be 'app' for Azure/Gunicorn to easily find it.
app = Flask(__name__)

# Configure CORS (use specific origins in production)
CORS(app, resources={r"/api/*": {"origins": "*", "supports_credentials": True}})

# --- Core LLM Logic ---

def execute_budget_buddy(query: str):
    """
    Skill: Budget Buddy
    Provides financial reasoning and cheaper alternatives using AI and Google Search.
    """
    if not model:
        raise Exception("AI model failed to initialize due to missing or invalid API key.")

    # System instruction to define the AI's persona
    system_prompt = (
        "You are Budget Buddy — the voice of financial reason. "
        "Your job is to help the user make smarter spending decisions. "
        "Suggest at least one cheaper or free alternative found via Google Search. "
        "Briefly explain why your choice is smarter. "
        "Keep it clever, casual, and concise — no lectures. "
        "If it’s already a good deal, say so with a friendly remark."
    )
    
    # Use Google Search grounding to find real, cheaper alternatives
    response = model.generate_content(
        query, 
        system_instruction=system_prompt,
        tools=[{"google_search": {}}]
    )
    return response.text

# --- Health Check Route ---

@app.route('/check', methods=['GET'])
def check():
    """
    Simple health check route used to confirm the backend server is running and accessible.
    """
    status_code = 200
    if not model:
        # Return 503 if the core dependency (AI model) failed to initialize
        status_code = 503
        message = "backend is running, but AI model failed to initialize."
    else:
        message = "backend is running"

    return jsonify({
        'status': 'ok' if status_code == 200 else 'error', 
        'message': message, 
        'model': MODEL_TO_USE
    }), status_code

# --- Main API Route ---

@app.route('/api/execute', methods=['POST'])
def execute():
    # 1. Input Validation
    if not model:
        return jsonify({'error': 'AI service not initialized. Check API key configuration.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload.'}), 400

    query = data.get('query')
    # Validate that query is a non-empty string
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'Missing or empty "query" field in the request.'}), 400
    
    # 2. Execution and Specific Error Handling
    try:
        result = execute_budget_buddy(query)
        
        return jsonify({'success': True, 'result': result})
        
    except APIError as e:
        # Handle specific Gemini API errors
        print(f"Gemini API Error: {e}")
        return jsonify({'error': f'AI Service Unavailable or request error: {e}'}), 503
        
    except Exception as e:
        # Catch all other unexpected errors
        print(f"Internal Server Error: {e}")
        return jsonify({'error': 'An unexpected internal server error occurred.'}), 500

