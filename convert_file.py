import requests
import json
import os

# --- Configuration ---
BASE_URL = "http://localhost:8000/api/v1"
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# --- Helper Functions ---

def read_file_content(file_path):
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def read_json_file(file_path):
    """Reads and parses a JSON file."""
    content = read_file_content(file_path)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {file_path}")
            return None
    return None

def convert_file_to_json(file_path, schema_path):
    """Converts a file to JSON using the MetaExtract API."""
    print(f"\n--- Starting conversion for {os.path.basename(file_path)} ---")

    # 1. Read the schema from the provided path
    schema = read_json_file(schema_path)
    if not schema:
        return

    # 2. Prepare the request for the /api/v1/convert endpoint
    url = f"{BASE_URL}/convert"
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'text/plain'),
        'schema': (None, json.dumps(schema), 'application/json')
    }

    # 3. Make the API call
    try:
        print("Sending request to the API...")
        response = requests.post(url, files=files)

        # 4. Handle the response
        if response.status_code == 200:
            result = response.json()
            print("\n--- Conversion Successful! ---")
            print("Strategy Used:", result.get('strategy_used'))
            print("Overall Confidence:", result.get('overall_confidence'))
            print("\nExtracted Data:")
            print(json.dumps(result.get('extracted_data', {}), indent=2))

            if result.get('low_confidence_fields'):
                print("\nLow Confidence Fields:", result['low_confidence_fields'])

        else:
            print(f"\n--- Error during conversion ---")
            print(f"Status Code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"\n--- API Request Failed ---")
        print(f"An error occurred: {e}")
        print("Please ensure the server is running. You can start it with 'python run_server.py'")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Test Case 1: GitHub Actions --- 
    github_input_path = os.path.join(current_dir, "testcases", "github actions sample input.md")
    github_schema_path = os.path.join(current_dir, "testcases", "github_actions_schema.json")
    convert_file_to_json(github_input_path, github_schema_path)

    # --- Test Case 2: Paper Citations ---
    paper_input_path = os.path.join(current_dir, "testcases", "NIPS-2017-attention-is-all-you-need-Bibtex.bib")
    paper_schema_path = os.path.join(current_dir, "testcases", "paper citations_schema.json")
    convert_file_to_json(paper_input_path, paper_schema_path)

    # --- Test Case 3: Resume --- 
    # Note: The resume schema is complex. We need a sample resume text file.
    # Creating a dummy resume file for demonstration.
    resume_text = """
    John Doe - Senior Software Engineer
    Email: john.doe@example.com
    Website: johndoe.dev

    Summary: Experienced software engineer with over 10 years of experience in Python and cloud computing.

    Work Experience:
    - Senior Software Engineer at Google (2018-Present)
      - Developed and maintained scalable microservices.
    - Software Engineer at Microsoft (2014-2018)
      - Worked on the Azure cloud platform.
    """
    resume_input_path = os.path.join(current_dir, "testcases", "sample_resume.txt")
    with open(resume_input_path, 'w') as f:
        f.write(resume_text)
    
    resume_schema_path = os.path.join(current_dir, "testcases", "convert your resume to this schema.json")
    convert_file_to_json(resume_input_path, resume_schema_path)

    # Clean up the dummy resume file
    os.remove(resume_input_path)
