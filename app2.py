from flask import Flask
import os
import logging
from flask import Flask, jsonify, request, render_template
from pypdf import PdfReader
from werkzeug.utils import secure_filename
import google.generativeai as genai
import yaml
import json
from flask_cors import CORS

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
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# Define raw_text globally
raw_text = ""


@app.route('/')
def index():
    """Upload PDF page."""
    return render_template('index.html')

@app.route('/process', methods=["POST"])
def process_resume():
    """Extract text from PDF and render form.html with extracted data."""
    global raw_text
    try:
        if 'pdf_doc' not in request.files:
            return "Error: No file uploaded", 400

        doc = request.files['pdf_doc']
        if not doc.filename.endswith('.pdf'):
            return "Error: Only PDF files are allowed", 400

        # Extract text from PDF
        raw_text = _read_file_from_memory(doc)
        if not raw_text:
            logging.error("No text extracted from PDF.")
            return "Error: No text extracted from PDF", 400
        # session["text"] = raw_text
        # Get structured JSON from Gemini
        parsed_json_string = ats_extractor(raw_text)
        if not parsed_json_string:
            logging.error("Gemini parsing failed.")
            return "Error: Failed to parse the resume.", 500

        logging.info(f"Raw JSON from Gemini:\n{parsed_json_string}")

        # Pass JSON data to form.html for user editing
        return render_template("form.html", json_data=parsed_json_string)

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return "An unexpected error occurred.", 500

@app.route('/submit', methods=["POST"])
def submit_details():
    """Handle the final form submission and redirect to ATS evaluation page."""
    try:
        submitted_data = request.form.to_dict()
        logging.info(f"User-submitted details: {submitted_data}")

        resume_json = json.dumps(submitted_data)

        # Render ATS evaluation form (ats.html)
        return render_template('ats.html', resume_json=resume_json)

    except Exception as e:
        logging.error(f"Error during submission: {e}")
        return "An unexpected error occurred.", 500

@app.route('/ats', methods=["POST"])
def ats_score():
    """Compute ATS score and render ats_result.html."""
    try:
        resume_json = request.form.get('resume_json')
        job_description = request.form.get('job_description')

        if not resume_json or not job_description:
            return "Error: Missing resume data or job description.", 400

        # Compute ATS score
        ats_result_raw = ats_score_extractor(resume_json, job_description)
        if not ats_result_raw:
            return "Error: Failed to compute ATS score.", 500

        try:
            ats_result_data = json.loads(ats_result_raw)
            ats_result_data["resume_json"] = json.loads(resume_json)  # Ensure it's always included
        except Exception as e:
            logging.error(f"Error parsing ATS result JSON: {e}")
            return "Error: Invalid ATS result format.", 500

        # Render ATS results page
        return render_template("ats_result.html", ats_result=ats_result_data)

    except Exception as e:
        logging.error(f"Error in ATS evaluation: {e}")
        return "An unexpected error occurred.", 500
    
@app.route('/generate_resume_html', methods=["POST"])
def generate_resume_html_endpoint():
    """
    Generate the final resume HTML using the candidate's resume JSON and
    additional missing skills from ATS.
    """
    global raw_text
    try:
        data = request.form.to_dict()
        if not data:
            return "Error: No data provided.", 400
        
        resume_json = json.loads(data.get("resume_json", "{}"))
        missing_skills = json.loads(data.get("missing_skills", "[]"))
        # raw_text = session.get("text")

        # Generate the resume content using Gemini AI
        resume_html = generate_resume_with_gemini(raw_text,resume_json, missing_skills)
        if not resume_html:
            return "Error: Failed to generate resume HTML.", 500

        return resume_html  # Directly return the generated resume HTML

    except Exception as e:
        logging.error(f"Error generating resume HTML: {e}")
        return "An unexpected error occurred.", 500


def generate_resume_with_gemini(raw_text2, resume_json, missing_skills):
    """
    Uses Gemini AI to fill the resume template with the candidate's data,
    integrates missing skills into the Technical Skills section, and utilizes
    the raw resume text to enhance the final HTML resume.
    """
    prompt = f"""
    You are an AI assistant that generates professional resumes. Given the structured resume JSON, 
    missing skills, and raw resume text, format the information into a complete, well-structured HTML resume.

    Resume Data (Structured JSON):
    {json.dumps(resume_json, indent=2)}

    Missing Skills:
    {json.dumps(missing_skills, indent=2)}

    Raw Resume Text (for reference):
    {raw_text2}

    Use the following HTML template and fill in the placeholders accurately.
    Ensure the "Technical Skills" section includes both the candidate's existing skills and the missing skills.
    Maintain a professional format.

    Template:
    {open("templates/generate_resume_html.html").read()}
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        resume_html = "".join(chunk.text for chunk in response)

        return resume_html.replace("```html", "").replace("```", "").strip()

    except Exception as e:
        logging.error(f"Error in generate_resume_with_gemini: {e}")
        return None




def _read_file_from_memory(file):
    """Reads and extracts text from an in-memory PDF file."""
    try:
        reader = PdfReader(file)
        data = ""
        for page_no in range(min(5, len(reader.pages))):  # Limit pages to save memory
            data += reader.pages[page_no].extract_text() or ""
        logging.info(f"Extracted Text (first 200 chars): {data[:200]}")
        return data
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""

def ats_extractor(resume_data):
    """Extracts structured JSON from resume text using Gemini API."""
    resume_data = resume_data.replace("%", "%%")  # Escape "%" to prevent format errors

    prompt = f"""
    You are an AI bot designed to parse resumes. Given the resume text provided below, extract key details and return them as valid JSON:

    {{
        "full_name": "",
        "email": "",
        "github": "",
        "linkedin": "",
        "employment": "",
        "technical_skills": [],
        "phone": "",
        "address": "",
        "profile": ""
    }}

    Ensure correct extraction with no extra text.

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

        return raw_result.replace("```json", "").replace("```", "").strip()

    except Exception as e:
        logging.error(f"Error in ats_extractor: {e}")
        return None

def ats_score_extractor(resume_json, job_description):
    """Computes ATS score by comparing resume JSON with job description."""
    prompt = f"""
    Compare the resume JSON below with the job description and calculate an ATS score (0-100). Also, list missing skills.

    Resume JSON:
    {resume_json}

    Job Description:
    {job_description}

    Return only JSON in this format:
    {{
        "ats_score": 0,
        "missing_skills": []
    }}
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, stream=True)
        raw_result = "".join(chunk.text for chunk in response)

        return raw_result.replace("```json", "").replace("```", "").strip()

    except Exception as e:
        logging.error(f"Error in ats_score_extractor: {e}")
        return None

if __name__ == "__main__":
    app.run(port=8000, debug=True)
