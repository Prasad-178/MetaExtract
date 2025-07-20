"""
Test script for Large Document Chunker

Demonstrates intelligent chunking of large documents with context preservation.
"""

import sys
sys.path.append('.')

from metaextract.core.document_chunker import LargeDocumentChunker
from metaextract.core.schema_analyzer import SchemaComplexityAnalyzer
import json

def load_schema(filename):
    """Load a JSON schema file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_text_file(filename):
    """Load a text file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def create_large_test_document():
    """Create a large test document for chunking"""
    return """
# Large Document Test

## Section 1: Personal Information
John Doe is a Senior Software Engineer based in San Francisco, CA. His email address is john.doe@techcorp.com and his phone number is (555) 123-4567. He has been working in the technology industry for over 10 years.

John graduated from the University of California, Berkeley in 2012 with a Bachelor's degree in Computer Science. His GPA was 3.8/4.0.

## Section 2: Professional Experience

### Tech Corp (2020-2023) - Senior Software Engineer
At Tech Corp, John led a team of 5 developers working on scalable web applications. Key achievements include:
- Developed a microservices architecture using React and Node.js
- Increased system performance by 40% through optimization
- Implemented CI/CD pipelines reducing deployment time by 60%
- Led migration from monolithic to microservices architecture

Technologies used: JavaScript, Python, React, Node.js, AWS, Docker, Kubernetes

### StartupXYZ (2018-2020) - Software Developer  
John worked on mobile applications and backend services:
- Built mobile applications using React Native for iOS and Android
- Developed RESTful APIs using Python and Django
- Implemented real-time features using WebSocket connections
- Worked with databases including PostgreSQL and MongoDB

### Previous Roles (2012-2018)
John started his career as a Junior Developer at various companies, gaining experience in:
- Web development using HTML, CSS, JavaScript
- Database design and optimization
- Version control with Git
- Agile development methodologies

## Section 3: Skills and Expertise

### Programming Languages
- JavaScript (Expert): 8+ years of experience
- Python (Advanced): 6+ years of experience  
- Java (Intermediate): 4+ years of experience
- TypeScript (Advanced): 3+ years of experience
- Go (Beginner): 1+ year of experience

### Frameworks and Technologies
- Frontend: React, Vue.js, Angular, HTML5, CSS3
- Backend: Node.js, Django, FastAPI, Express.js
- Mobile: React Native, Flutter
- Databases: PostgreSQL, MongoDB, Redis, MySQL
- Cloud: AWS, Google Cloud, Azure
- DevOps: Docker, Kubernetes, Jenkins, GitLab CI

### Soft Skills
- Team Leadership
- Project Management
- Technical Writing
- Code Review
- Mentoring

## Section 4: Projects

### Project Alpha - E-commerce Platform
Duration: 2022-2023
Role: Technical Lead
Team Size: 8 developers

Description: Led the development of a large-scale e-commerce platform serving 1M+ users.

Key Features:
- User authentication and authorization
- Product catalog with search functionality
- Shopping cart and checkout process
- Payment integration with Stripe and PayPal
- Order management system
- Admin dashboard for inventory management

Technologies: React, Node.js, PostgreSQL, Redis, AWS

Results:
- Successfully launched to 100,000+ users in first month
- Achieved 99.9% uptime SLA
- Processing $10M+ in transactions monthly

### Project Beta - Data Analytics Dashboard
Duration: 2021-2022
Role: Full-Stack Developer
Team Size: 4 developers

Description: Built a real-time data analytics dashboard for business intelligence.

Features:
- Real-time data visualization using D3.js
- Custom reporting with filters and exports
- Role-based access control
- API integrations with multiple data sources
- Automated report generation

Technologies: Vue.js, Python, Flask, ClickHouse, Docker

Results:
- Reduced report generation time from hours to minutes
- Improved decision-making speed by 50%
- Adopted by 200+ internal users

## Section 5: Education and Certifications

### University of California, Berkeley (2008-2012)
Bachelor of Science in Computer Science
GPA: 3.8/4.0

Relevant Coursework:
- Data Structures and Algorithms
- Database Systems
- Computer Networks
- Software Engineering
- Machine Learning
- Computer Graphics

### Certifications
- AWS Certified Solutions Architect (2021)
- Google Cloud Professional Developer (2020)
- Certified Kubernetes Administrator (2022)
- Scrum Master Certification (2019)

## Section 6: Contact Information

Email: john.doe@techcorp.com
Phone: (555) 123-4567
LinkedIn: https://linkedin.com/in/johndoe
GitHub: https://github.com/johndoe
Personal Website: https://johndoe.dev

Address:
123 Tech Street
San Francisco, CA 94105
United States

## Additional Information

John is passionate about open source software and has contributed to several projects:
- React Router (5+ contributions)
- Node.js (2+ contributions)  
- Django Rest Framework (3+ contributions)

He regularly speaks at technology conferences and has given talks on:
- Microservices Architecture Best Practices
- Performance Optimization in React Applications
- Building Scalable APIs with Node.js

John is also involved in mentoring junior developers and has mentored 10+ developers throughout his career.

## Table Example

| Technology | Experience | Projects |
|------------|------------|----------|
| JavaScript | 8 years    | 15+      |
| Python     | 6 years    | 12+      |
| React      | 5 years    | 10+      |
| Node.js    | 6 years    | 8+       |
| AWS        | 4 years    | 6+       |

## JSON Example

{
  "name": "John Doe",
  "title": "Senior Software Engineer", 
  "email": "john.doe@techcorp.com",
  "experience_years": 10,
  "skills": ["JavaScript", "Python", "React", "Node.js", "AWS"],
  "current_company": "Tech Corp",
  "location": "San Francisco, CA"
}
"""

def test_document_chunking():
    """Test document chunking with different scenarios"""
    chunker = LargeDocumentChunker(max_chunk_size=2000, overlap_size=200)
    
    print("MetaExtract: Large Document Chunker Test")
    print("=" * 50)
    
    # Test 1: Large document without schema
    print("\n📄 Test 1: Large Document (No Schema)")
    print("-" * 30)
    
    large_doc = create_large_test_document()
    print(f"Document length: {len(large_doc)} characters")
    
    plan = chunker.chunk_document(large_doc, document_type="markdown")
    
    print(f"Chunks created: {plan.total_chunks}")
    print(f"Total tokens: {plan.total_tokens:,}")
    print(f"Requires overlap handling: {plan.requires_overlap_handling}")
    
    # Show priority distribution
    print(f"\nPriority Distribution:")
    for priority, chunk_ids in plan.priority_groups.items():
        if chunk_ids:
            print(f"  {priority.name}: {len(chunk_ids)} chunks")
    
    # Show first few chunks
    print(f"\nFirst 3 chunks:")
    for i, chunk_id in enumerate(plan.processing_order[:3], 1):
        chunk = plan.chunks[chunk_id]
        content_preview = chunk.content[:100].replace('\n', ' ')
        print(f"  {i}. {chunk_id} ({chunk.chunk_type.value}, {chunk.priority.name})")
        print(f"     Tokens: {chunk.estimated_tokens}, Content: {content_preview}...")
    
    # Test 2: Document with schema guidance
    print("\n\n📄 Test 2: Document with Schema Guidance")
    print("-" * 40)
    
    # Load resume schema
    schema = load_schema("testcases/convert your resume to this schema.json")
    plan_with_schema = chunker.chunk_document(large_doc, schema=schema, document_type="markdown")
    
    print(f"Chunks created: {plan_with_schema.total_chunks}")
    print(f"Total tokens: {plan_with_schema.total_tokens:,}")
    
    # Show how schema guidance affected chunking
    print(f"\nSchema-guided priority distribution:")
    for priority, chunk_ids in plan_with_schema.priority_groups.items():
        if chunk_ids:
            total_tokens = sum(plan_with_schema.chunks[cid].estimated_tokens for cid in chunk_ids)
            print(f"  {priority.name}: {len(chunk_ids)} chunks ({total_tokens:,} tokens)")
    
    # Show chunks with high relevance
    high_relevance_chunks = []
    for chunk in plan_with_schema.chunks.values():
        if chunk.confidence_score > 0.7:
            high_relevance_chunks.append(chunk)
    
    print(f"\nHigh relevance chunks ({len(high_relevance_chunks)}):")
    for chunk in high_relevance_chunks[:3]:
        content_preview = chunk.content[:100].replace('\n', ' ')
        print(f"  {chunk.chunk_id}: confidence={chunk.confidence_score:.2f}")
        print(f"    Content: {content_preview}...")

def test_github_actions_input():
    """Test chunking with the provided GitHub Actions sample"""
    print("\n\n📄 Test 3: GitHub Actions Sample Input")
    print("-" * 40)
    
    chunker = LargeDocumentChunker(max_chunk_size=1500)
    
    # Load GitHub Actions sample and schema
    github_input = load_text_file("testcases/github actions sample input.md")
    github_schema = load_schema("testcases/github_actions_schema.json")
    
    print(f"Input length: {len(github_input)} characters")
    
    plan = chunker.chunk_document(github_input, schema=github_schema, document_type="markdown")
    
    print(f"Chunks created: {plan.total_chunks}")
    print(f"Total tokens: {plan.total_tokens:,}")
    
    # Show chunking summary
    summary = chunker.get_chunking_summary(plan)
    print(f"\n{summary}")
    
    # Show chunk details
    print(f"\nDetailed Chunk Analysis:")
    for chunk_id, chunk in plan.chunks.items():
        print(f"  🔹 {chunk_id}:")
        print(f"     Type: {chunk.chunk_type.value}")
        print(f"     Priority: {chunk.priority.name}")
        print(f"     Tokens: {chunk.estimated_tokens}")
        print(f"     Confidence: {chunk.confidence_score:.2f}")
        print(f"     Position: {chunk.start_position}-{chunk.end_position}")
        
        if chunk.section_context:
            print(f"     Section: {chunk.section_context}")
        
        if chunk.related_chunk_ids:
            print(f"     Related: {', '.join(chunk.related_chunk_ids[:3])}")
        
        content_preview = chunk.content[:150].replace('\n', ' ')
        print(f"     Content: {content_preview}...")
        print()

def test_performance_with_large_input():
    """Test performance with a very large input"""
    print("\n\n📄 Test 4: Performance with Large Input")
    print("-" * 40)
    
    chunker = LargeDocumentChunker(max_chunk_size=3000)
    
    # Create a very large document by repeating content
    base_doc = create_large_test_document()
    very_large_doc = base_doc * 20  # Simulate a very large document
    
    print(f"Very large document: {len(very_large_doc):,} characters")
    print(f"Estimated size: {len(very_large_doc) / (1024*1024):.2f} MB")
    
    import time
    start_time = time.time()
    
    plan = chunker.chunk_document(very_large_doc, document_type="markdown")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Chunks created: {plan.total_chunks}")
    print(f"Total tokens: {plan.total_tokens:,}")
    print(f"Average chunk size: {plan.total_tokens // plan.total_chunks} tokens")
    print(f"Processing speed: {len(very_large_doc) / processing_time:,.0f} chars/second")

if __name__ == "__main__":
    test_document_chunking()
    test_github_actions_input()
    test_performance_with_large_input() 