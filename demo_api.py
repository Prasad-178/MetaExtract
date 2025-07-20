"""
Demonstration script for MetaExtract API.
Shows how to use the API with different schemas and input types.
"""
import asyncio
import json
import httpx
import time
from typing import Dict, Any


class MetaExtractAPIDemo:
    """Demo client for MetaExtract API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def check_health(self):
        """Check API health."""
        print("🔍 Checking API health...")
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ API is healthy - Version: {health['version']}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to API: {e}")
            return False
    
    def load_schema(self, schema_file: str) -> Dict[str, Any]:
        """Load a JSON schema from file."""
        try:
            with open(schema_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load schema {schema_file}: {e}")
            return {}
    
    def load_sample_text(self, text_file: str) -> str:
        """Load sample text from file."""
        try:
            with open(text_file, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Failed to load text {text_file}: {e}")
            return ""
    
    async def analyze_schema(self, schema: Dict[str, Any], schema_name: str):
        """Analyze schema complexity."""
        print(f"\n📊 Analyzing {schema_name} schema complexity...")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/analyze-schema",
                json={"schema": schema}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Schema Analysis for {schema_name}:")
                print(f"   📈 Complexity: {result['complexity_metrics']['complexity_level']}")
                print(f"   🎯 Recommended Strategy: {result['recommended_strategy']}")
                print(f"   ⏱️ Estimated Time: {result['estimated_processing_time']:.2f}s")
                print(f"   💾 Memory Estimate: {result['resource_requirements']['memory_estimate']}")
                return result
            else:
                print(f"❌ Schema analysis failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Schema analysis error: {e}")
            return None
    
    async def extract_from_text(self, text: str, schema: Dict[str, Any], strategy: str = "auto"):
        """Perform text extraction."""
        print(f"\n🔍 Extracting data using {strategy} strategy...")
        print(f"   📝 Input length: {len(text)} characters")
        
        try:
            start_time = time.time()
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/extract",
                json={
                    "input_text": text,
                    "schema": schema,
                    "strategy": strategy,
                    "confidence_threshold": 0.7
                }
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Extraction completed in {end_time - start_time:.2f}s")
                print(f"   📋 Strategy Used: {result['strategy_used']}")
                print(f"   🎯 Overall Confidence: {result['overall_confidence']:.2f}")
                print(f"   📊 Chunks Processed: {result['chunk_count']}")
                print(f"   🤖 Agents Used: {result['agent_count']}")
                
                if result['validation_errors']:
                    print(f"   ⚠️ Validation Errors: {len(result['validation_errors'])}")
                
                if result['low_confidence_fields']:
                    print(f"   🔍 Low Confidence Fields: {result['low_confidence_fields']}")
                
                return result
            else:
                print(f"❌ Extraction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return None
    
    async def test_async_extraction(self, text: str, schema: Dict[str, Any]):
        """Test async extraction endpoint."""
        print(f"\n🔄 Testing async extraction...")
        
        try:
            # Start async job
            response = await self.client.post(
                f"{self.base_url}/api/v1/extract/async",
                json={
                    "input_text": text,
                    "schema": schema,
                    "strategy": "auto"
                }
            )
            
            if response.status_code == 200:
                job_info = response.json()
                job_id = job_info['job_id']
                print(f"✅ Async job started: {job_id}")
                
                # Poll for completion
                while True:
                    status_response = await self.client.get(
                        f"{self.base_url}/api/v1/extract/status/{job_id}"
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"   📈 Progress: {status['progress']*100:.1f}% - Status: {status['status']}")
                        
                        if status['status'] in ['completed', 'failed']:
                            if status['status'] == 'completed':
                                print(f"✅ Async extraction completed!")
                                return status['result']
                            else:
                                print(f"❌ Async extraction failed")
                                return None
                    
                    await asyncio.sleep(1)
            else:
                print(f"❌ Failed to start async job: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Async extraction error: {e}")
            return None
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("🚀 MetaExtract API Demonstration")
        print("=" * 50)
        
        # Check API health
        if not await self.check_health():
            return
        
        # Load schemas and sample data
        schemas = {
            "Resume": self.load_schema("testcases/convert your resume to this schema.json"),
            "GitHub Actions": self.load_schema("testcases/github_actions_schema.json"),
            "Paper Citations": self.load_schema("testcases/paper citations_schema.json")
        }
        
        # Sample resume text
        resume_text = """
        John Doe
        Senior Software Engineer
        Email: john.doe@email.com
        Phone: (555) 123-4567
        Location: San Francisco, CA
        
        EXPERIENCE:
        - Senior Software Engineer at TechCorp (2020-2023)
          * Led development of microservices architecture
          * Managed team of 5 developers
          * Technologies: Python, React, AWS, Docker
        
        - Software Engineer at StartupXYZ (2018-2020)
          * Built scalable web applications
          * Implemented CI/CD pipelines
          * Technologies: Node.js, MongoDB, Kubernetes
        
        EDUCATION:
        - Master of Science in Computer Science, Stanford University (2018)
        - Bachelor of Science in Computer Science, UC Berkeley (2016)
        
        SKILLS:
        - Programming: Python, JavaScript, Java, Go
        - Cloud: AWS, GCP, Azure
        - Databases: PostgreSQL, MongoDB, Redis
        - DevOps: Docker, Kubernetes, Jenkins
        """
        
        # Load GitHub Actions sample
        github_actions_text = self.load_sample_text("testcases/github actions sample input.md")
        
        # Demo 1: Schema Analysis
        print("\n" + "="*50)
        print("📊 SCHEMA ANALYSIS DEMO")
        print("="*50)
        
        for name, schema in schemas.items():
            if schema:
                await self.analyze_schema(schema, name)
        
        # Demo 2: Resume Extraction
        print("\n" + "="*50)
        print("👤 RESUME EXTRACTION DEMO")
        print("="*50)
        
        if schemas["Resume"]:
            result = await self.extract_from_text(resume_text, schemas["Resume"], "auto")
            if result and result['extracted_data']:
                print("\n📄 Extracted Resume Data:")
                self.print_json(result['extracted_data'])
        
        # Demo 3: GitHub Actions Extraction
        print("\n" + "="*50)
        print("⚙️ GITHUB ACTIONS EXTRACTION DEMO")
        print("="*50)
        
        if schemas["GitHub Actions"] and github_actions_text:
            result = await self.extract_from_text(github_actions_text, schemas["GitHub Actions"], "hierarchical_chunking")
            if result and result['extracted_data']:
                print("\n📄 Extracted GitHub Actions Data:")
                self.print_json(result['extracted_data'])
        
        # Demo 4: Strategy Comparison
        print("\n" + "="*50)
        print("🔄 STRATEGY COMPARISON DEMO")
        print("="*50)
        
        if schemas["Resume"]:
            strategies = ["simple_prompt", "enhanced_prompt", "multi_agent_parallel"]
            
            for strategy in strategies:
                print(f"\n🧪 Testing {strategy} strategy:")
                result = await self.extract_from_text(resume_text[:500], schemas["Resume"], strategy)
                if result:
                    print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")
                    print(f"   🎯 Confidence: {result['overall_confidence']:.2f}")
        
        # Demo 5: Async Processing
        print("\n" + "="*50)
        print("🔄 ASYNC PROCESSING DEMO")
        print("="*50)
        
        if schemas["Resume"]:
            await self.test_async_extraction(resume_text, schemas["Resume"])
        
        print("\n🎉 Demo completed!")
        await self.client.aclose()
    
    def print_json(self, data: Any, max_depth: int = 3, current_depth: int = 0):
        """Print JSON data with limited depth for readability."""
        if current_depth > max_depth:
            print("   " * current_depth + "... (truncated)")
            return
        
        if isinstance(data, dict):
            for key, value in list(data.items())[:5]:  # Limit to first 5 items
                if isinstance(value, (dict, list)):
                    print("   " * current_depth + f"{key}:")
                    self.print_json(value, max_depth, current_depth + 1)
                else:
                    print("   " * current_depth + f"{key}: {value}")
            if len(data) > 5:
                print("   " * current_depth + "... (more fields)")
        
        elif isinstance(data, list):
            for i, item in enumerate(data[:3]):  # Limit to first 3 items
                print("   " * current_depth + f"[{i}]:")
                self.print_json(item, max_depth, current_depth + 1)
            if len(data) > 3:
                print("   " * current_depth + "... (more items)")
        else:
            print("   " * current_depth + str(data))


async def main():
    """Main demo function."""
    demo = MetaExtractAPIDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("Starting MetaExtract API Demo...")
    print("Make sure the API server is running on http://localhost:8000")
    print("You can start it with: python -m api.main")
    print()
    
    asyncio.run(main()) 