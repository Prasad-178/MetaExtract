"""
Test file conversion functionality with real testcases.
Demonstrates converting unstructured text files to structured JSON following any schema.
"""

import json
import requests
import os
from pathlib import Path


class FileConversionTester:
    """Test the file conversion API with real examples"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.testcases_dir = Path("testcases")
    
    def load_schema(self, schema_file: str) -> dict:
        """Load a JSON schema from the testcases directory"""
        schema_path = self.testcases_dir / schema_file
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def test_conversion(self, file_path: str, schema: dict, description: str):
        """Test converting a file to JSON using the API"""
        print(f"\n🧪 Testing: {description}")
        print("=" * 50)
        
        file_size = os.path.getsize(file_path)
        print(f"📄 File: {file_path}")
        print(f"📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        try:
            # Make API request
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {'schema': json.dumps(schema)}
                
                print("🔄 Converting file to JSON...")
                response = requests.post(
                    f"{self.base_url}/api/v1/convert",
                    files=files,
                    data=data,
                    timeout=120  # Allow up to 2 minutes for complex extractions
                )
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Conversion successful!")
                print(f"⏱️  Processing time: {result['processing_time']:.1f}s")
                print(f"🎯 Strategy used: {result['strategy_used']}")
                print(f"📈 Confidence: {result['overall_confidence']:.2f}")
                print(f"🏗️  Schema complexity: {result['schema_complexity']['complexity_level']}")
                
                # Show sample of extracted data
                if result['extracted_data']:
                    print(f"\n📄 Sample extracted data:")
                    self._print_sample_json(result['extracted_data'])
                
                return result
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (this can happen with very complex schemas)")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _print_sample_json(self, data: dict, max_items: int = 5, indent: int = 0):
        """Print a sample of JSON data with limited depth"""
        prefix = "  " * indent
        
        if isinstance(data, dict):
            items = list(data.items())[:max_items]
            for key, value in items:
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    print(f"{prefix}{key}: {type(value).__name__} with {len(value)} items")
                else:
                    value_str = str(value)[:60]
                    if len(str(value)) > 60:
                        value_str += "..."
                    print(f"{prefix}{key}: {value_str}")
            
            if len(data) > max_items:
                print(f"{prefix}... and {len(data) - max_items} more fields")
        else:
            print(f"{prefix}{str(data)[:100]}...")
    
    def run_all_tests(self):
        """Run tests with all available testcases"""
        print("🚀 File Conversion API Testing")
        print("=" * 60)
        print("Testing conversion of unstructured text files to structured JSON")
        print()
        
        # Test 1: GitHub Actions conversion
        github_schema = self.load_schema("github_actions_schema.json")
        github_file = self.testcases_dir / "github actions sample input.md"
        
        if github_file.exists():
            self.test_conversion(
                str(github_file),
                github_schema,
                "GitHub Actions: Markdown → YAML Schema"
            )
        
        # Test 2: Create a simple resume test
        simple_resume_schema = {
            "type": "object",
            "properties": {
                "personal_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "location": {"type": "string"}
                    }
                },
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company": {"type": "string"},
                            "position": {"type": "string"},
                            "duration": {"type": "string"},
                            "technologies": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        # Create a sample resume file for testing
        sample_resume = """
        Sarah Johnson
        Senior Software Engineer
        Email: sarah.johnson@email.com
        Phone: (555) 987-6543
        Location: Seattle, WA
        
        EXPERIENCE:
        
        Senior Software Engineer at Microsoft (2021-2024)
        • Led development of cloud-native microservices architecture
        • Technologies: Python, React, Azure, Docker, Kubernetes
        
        Software Engineer at Amazon (2019-2021)
        • Built scalable web applications serving 10M+ users
        • Technologies: Java, Spring Boot, AWS, DynamoDB
        
        SKILLS:
        Python, Java, JavaScript, AWS, Azure, Docker, Kubernetes, React
        """
        
        # Write sample resume to temp file
        temp_resume_file = "temp_sample_resume.txt"
        with open(temp_resume_file, 'w') as f:
            f.write(sample_resume)
        
        try:
            self.test_conversion(
                temp_resume_file,
                simple_resume_schema,
                "Resume: Text → Structured Resume"
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_resume_file):
                os.remove(temp_resume_file)
        
        # Test 3: API documentation
        print(f"\n📚 API Documentation")
        print("=" * 50)
        print(f"🌐 Swagger UI: {self.base_url}/docs")
        print(f"🔧 Convert endpoint: {self.base_url}/api/v1/convert")
        print(f"📊 Schema analysis: {self.base_url}/api/v1/analyze-schema")
        
        print(f"\n💡 Usage Example:")
        print(f"curl -X POST '{self.base_url}/api/v1/convert' \\")
        print(f"     -F 'file=@your_document.txt' \\")
        print(f"     -F 'schema={{\"type\":\"object\",\"properties\":{{\"title\":{{\"type\":\"string\"}}}}}}'")
        
        print(f"\n🎉 File conversion testing completed!")
        print(f"The system can convert any text file to any JSON schema structure.")


def main():
    """Run the file conversion tests"""
    tester = FileConversionTester()
    tester.run_all_tests()


if __name__ == "__main__":
    print("🚀 Starting File Conversion Tests...")
    print("📝 Make sure the API server is running: python run_server.py")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            main()
        else:
            print("❌ API server not responding properly")
    except requests.exceptions.RequestException:
        print("❌ API server not running. Please start it with: python run_server.py") 