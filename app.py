import os
import logging
from flask import Flask, jsonify, request, render_template
from pypdf import PdfReader
from werkzeug.utils import secure_filename
import google.generativeai as genai
import yaml
import json
from flask_cors import CORS
# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure paths
UPLOAD_PATH = os.path.join(os.getcwd(), "__DATA__")
os.makedirs(UPLOAD_PATH, exist_ok=True)

# Load API key from configuration file
CONFIG_PATH = r"config.yaml"
with open(CONFIG_PATH) as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    api_key = data['GEMINI_API_KEY']

# Configure the Gemini API client
genai.configure(api_key=api_key)

app = Flask(__name__)
CORS(app, resources={
     r"/*": {"origins": "https://nextgen-navigator.vercel.app"}})

@app.route('/')
def index():
    """Upload PDF page."""
    return render_template('index.html')


@app.route('/process', methods=["POST"])
def process_resume():
    """Extract text from PDF without saving it locally."""
    try:
        # 1. Check if file is uploaded
        if 'pdf_doc' not in request.files:
            return "Error: No file uploaded", 400

        doc = request.files['pdf_doc']
        if not doc.filename.endswith('.pdf'):
            return "Error: Only PDF files are allowed", 400

        # 2. Read file from memory (stream)
        raw_text = _read_file_from_memory(doc)
        if not raw_text:
            logging.error("No text extracted from PDF.")
            return "Error: No text extracted from PDF", 400

        # 3. Get raw JSON string from Gemini
        parsed_json_string = ats_extractor(raw_text)
        if not parsed_json_string:
            logging.error("Gemini parsing failed.")
            return "Error: Failed to parse the resume.", 500

        logging.info(f"Raw JSON from Gemini:\n{parsed_json_string}")

        return parsed_json_string

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return "An unexpected error occurred.", 500


def _read_file_from_memory(file):
    """Reads and extracts text from an in-memory PDF file."""
    try:
        reader = PdfReader(file)  # Read directly from memory
        data = ""
        for page_no in range(min(5, len(reader.pages))):  # Limit pages to save memory
            data += reader.pages[page_no].extract_text() or ""
        logging.info(f"Extracted Text (first 200 chars): {data[:200]}")
        return data
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""

@app.route('/submit', methods=["POST"])
def submit_details():
    """Handle the final form submission (if needed)."""
    try:
        submitted_data = request.form.to_dict()
        logging.info(f"User-submitted details: {submitted_data}")
        # You might want to redirect the user to the ATS evaluation page here.
        # For example, pass the resume JSON as a query parameter:
        resume_json = json.dumps(submitted_data)  # or however you want to pass it
        # return render_template('ats.html', resume_json=resume_json)
        return resume_json
    
    except Exception as e:
        logging.error(f"Error during submission: {e}")
        return "An unexpected error occurred.", 500

# def _read_file_from_path(path):
#     """Reads and extracts text from a PDF file."""
#     try:
#         reader = PdfReader(path)
#         data = ""
#         for page_no in range(len(reader.pages)):
#             data += reader.pages[page_no].extract_text() or ""
#         logging.info(f"Extracted Text (first 200 chars): {data[:200]}")
#         return data
#     except Exception as e:
#         logging.error(f"Error reading PDF file: {e}")
#         return ""

def ats_extractor(resume_data):
    """
    Query Gemini with instructions to return JSON ONLY.
    Returns the raw JSON text as a string (no python parsing).
    """
    prompt = f"""
    You are an AI bot designed to parse resumes. Given the resume below, extract:
    1. Full name
    2. Email ID
    3. GitHub portfolio
    4. LinkedIn ID
    5. Employment details
    6. Technical skills
    7. Soft skills

    Return valid JSON only, with no extra text outside the JSON object.

    Resume:
    {resume_data}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        raw_result = "".join(chunk.text for chunk in response)

        if not raw_result.strip():
            logging.error("Error: Gemini response is empty.")
            return None

        # Minimal cleanup for code fences:
        cleaned_result = (
            raw_result.replace("```json", "")
                      .replace("```", "")
                      .strip()
        )
        return cleaned_result
    except Exception as e:
        logging.error(f"Error in ats_extractor: {e}")
        return None

# -------------------------------
# New endpoint and function for ATS scoring
# -------------------------------


@app.route('/ats', methods=["POST"])
def ats_score():

    # Get resume JSON and job description from the POST request
    data = request.get_json()  

    resume_json = data.get('resume_json')
    job_description = data.get('job_description')
    print(resume_json)

    if not resume_json or not job_description:
        return "Error: Missing resume data or job description.", 400

    # Process the ATS score using the resume and job description
    ats_result_raw = ats_score_extractor(resume_json, job_description)
    if not ats_result_raw:
        return "Error: Failed to compute ATS score.", 500

    try:
        ats_result_data = json.loads(ats_result_raw)
    except Exception as e:
        logging.error(f"Error parsing ATS result JSON: {e}")
        return "Error: Invalid ATS result format.", 500

    return jsonify(ats_result_data)


def ats_score_extractor(resume_json, job_description):
    """
    Query Gemini to compute an ATS score by comparing the resume JSON with the job description.
    Returns a JSON string that includes an "ats_score" (number) and "missing_skills" (array of strings).
    """
    prompt = f"""
    You are an AI assistant that calculates an ATS (Applicant Tracking System) score based on a candidate's resume and a job description.
    The candidate's resume is provided as valid JSON:
    {resume_json}
    The job description is:
    {job_description}
    Compare the candidate's resume details with the job description requirements.
    Calculate an ATS score between 0 and 100, where 100 represents a perfect match.
    Also, identify and list any missing skills or qualifications that are required in the job description but are not present in the resume JSON.
    Return a JSON object with two keys: "ats_score" (a number) and "missing_skills" (an array of strings with keywords only).
    Return only valid JSON with no additional text.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        raw_result = "".join(chunk.text for chunk in response)
        cleaned_result = (
            raw_result.replace("```json", "")
                      .replace("```", "")
                      .strip()
        )
        return cleaned_result
    except Exception as e:
        logging.error(f"Error in ats_score_extractor: {e}")
        return None
if __name__ == "__main__":
    app.run(port=8000, debug=True)
