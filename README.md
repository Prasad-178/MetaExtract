# MetaExtract: Intelligent Unstructured Text to JSON Conversion

MetaExtract is a sophisticated, AI-powered system designed to convert unstructured text from various sources into a structured JSON format. It strictly adheres to a user-provided JSON schema, intelligently adapting its extraction strategy based on the complexity of the schema and the size of the input data.

## Key Features

*   **Advanced Schema-Guided Extraction:** Accurately extracts data from unstructured text (including `.txt`, `.md`, `.csv`, and more) and maps it to a complex, deeply nested JSON schema.
*   **Dynamic Strategy Selection:** The system automatically analyzes the input schema and document to choose the most effective extraction strategy from the following options:
    *   **Simple:** A direct, single-call approach for small documents and simple schemas.
    *   **Chunked:** An intelligent chunking mechanism for large documents, ensuring all data is processed without exceeding context limits.
    *   **Hierarchical:** A divide-and-conquer approach that breaks down complex schemas into manageable sections for more accurate and reliable extraction.
*   **Robust and Scalable:** Built with a modular architecture, MetaExtract is capable of handling large documents (up to 10MB) and highly complex schemas with multiple levels of nesting.
*   **Comprehensive API:** A well-documented FastAPI application exposes the full power of the extraction engine through a clean and intuitive REST API.
*   **Confidence and Validation:** Provides detailed confidence scores for each extracted field and flags low-confidence fields for human review, ensuring data quality and reliability.
*   **Asynchronous Processing:** For time-intensive extractions, the API provides an asynchronous endpoint, allowing users to submit jobs and check their status without blocking.

## Getting Started

### Prerequisites

*   Python 3.10+
*   An OpenAI API Key

### 1. Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/metaextract.git
cd metaextract
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project's root directory and add your OpenAI API key:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. Running the API Server

Start the FastAPI server with the following command:

```bash
python run_server.py
```

The API documentation will be available at `http://localhost:8000/docs`.

## API Usage

The MetaExtract API provides a comprehensive set of endpoints for interacting with the system.

### Health Check

*   **GET** `/api/v1/health`

    Returns the current status of the API and its components.

### Schema Analysis

*   **POST** `/api/v1/analyze-schema`

    Analyzes a given JSON schema and returns detailed complexity metrics, a recommended extraction strategy, and estimated processing time. This endpoint does not require an OpenAI API key.

    **Example:**

    ```bash
    curl -X POST "http://localhost:8000/api/v1/analyze-schema" \
      -H "Content-Type: application/json" \
      -d '{"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}'
    ```

### Text-Based Extraction

*   **POST** `/api/v1/extract`

    Extracts structured data from a raw text string based on the provided JSON schema.

    **Example:**

    ```bash
    curl -X POST "http://localhost:8000/api/v1/extract" \
      -H "Content-Type: application/json" \
      -d '{
        "input_text": "John Doe is a software engineer at Google.",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "title": {"type": "string"},
            "company": {"type": "string"}
          }
        },
        "strategy": "auto"
      }'
    ```

### File-Based Extraction

*   **POST** `/api/v1/convert`

    A simplified endpoint for converting a file directly to structured JSON. The system automatically selects the best extraction strategy.

    **Example:**

    ```bash
    curl -X POST "http://localhost:8000/api/v1/convert" \
      -F "file=@/path/to/your/document.txt" \
      -F 'schema={"type": "object", "properties": {"name": {"type": "string"}}}'
    ```

*   **POST** `/api/v1/extract/file`

    A more advanced endpoint for file-based extraction that allows you to specify the extraction strategy.

### Asynchronous Extraction

*   **POST** `/api/v1/extract/async`

    Initiates an asynchronous extraction job and returns a `job_id`.

*   **GET** `/api/v1/extract/status/{job_id}`

    Retrieves the status and results of an asynchronous extraction job.

## Demonstrations and Testing

The project includes several scripts to demonstrate its capabilities and run a comprehensive test suite.

*   **Schema Analysis and Strategy Selection Demo:**

    ```bash
    python simplified_demo.py
    ```

*   **Complete System Test:**

    ```bash
    python test_complete_system.py
    ```

*   **File Conversion Test:**

    ```bash
    python test_file_conversion.py
    ```

*   **File Conversion Script:**

    ```bash
    python convert_file.py
    ```

## Architecture

The core of the system is the `SimplifiedMetaExtract` class, which contains the logic for schema analysis, strategy selection, and data extraction. This class is integrated with a FastAPI application that exposes its functionality through a REST API. The system is designed to be modular and extensible, allowing for the future addition of new extraction strategies and models.

## Project Structure

```
metaforms-assignment/
├── api/
│   ├── config.py               # API and application configuration
│   ├── main.py                 # FastAPI application entrypoint
│   ├── models.py               # Pydantic models for API requests and responses
│   └── routes.py               # API endpoint definitions
├── metaextract/
│   └── simplified_extractor.py # Core extraction engine
├── testcases/                  # Sample schemas and data for testing
├── convert_file.py             # Script to convert a file to JSON using the API
├── run_server.py               # Script to start the API server
├── simplified_demo.py          # Demonstrates schema analysis and strategy selection
├── test_complete_system.py     # Comprehensive test suite
├── test_file_conversion.py     # Tests the file conversion functionality
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
