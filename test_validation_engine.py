"""
Test script for Validation & Confidence Engine

Demonstrates validation and confidence assessment for extracted data.
"""

import sys
sys.path.append('.')

from metaextract.core.validation_engine import ValidationEngine, ValidationLevel
import json

def load_schema(filename):
    """Load a JSON schema file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_validation_scenarios():
    """Test various validation scenarios"""
    validator = ValidationEngine(validation_level=ValidationLevel.STRICT)
    
    print("MetaExtract: Validation & Confidence Engine Test")
    print("=" * 60)
    
    # Test 1: Perfect data
    print("\n🧪 Test 1: Perfect Valid Data")
    print("-" * 30)
    
    perfect_data = {
        "name": "John Doe",
        "email": "john.doe@techcorp.com",
        "phone": "(555) 123-4567",
        "age": 30,
        "skills": ["JavaScript", "Python", "React"],
        "active": True
    }
    
    simple_schema = {
        "type": "object",
        "required": ["name", "email"],
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "phone": {"type": "string"},
            "age": {"type": "integer", "minimum": 18, "maximum": 100},
            "skills": {"type": "array", "items": {"type": "string"}},
            "active": {"type": "boolean"}
        }
    }
    
    source_text = "John Doe is 30 years old. His email is john.doe@techcorp.com and phone is (555) 123-4567. He knows JavaScript, Python, and React."
    
    result = validator.validate_extraction(perfect_data, simple_schema, source_text)
    
    print(f"Validation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Overall Confidence: {result.overall_confidence:.2f}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Low Confidence Fields: {len(result.low_confidence_fields)}")
    print(f"  Human Review Required: {result.human_review_required}")
    
    print(f"\nField Confidence Breakdown:")
    for field_path, confidence in result.field_confidences.items():
        print(f"  {field_path}: {confidence.confidence_score:.2f} ({confidence.confidence_level.name})")
    
    # Test 2: Data with violations
    print("\n🧪 Test 2: Data with Schema Violations")
    print("-" * 40)
    
    invalid_data = {
        "name": "",  # Empty required field
        "email": "not-an-email",  # Invalid format
        "age": "thirty",  # Wrong type
        "skills": "JavaScript, Python",  # Wrong type (should be array)
        "location": "San Francisco"  # Additional property
    }
    
    result2 = validator.validate_extraction(invalid_data, simple_schema, source_text)
    
    print(f"Validation Result:")
    print(f"  Valid: {result2.is_valid}")
    print(f"  Overall Confidence: {result2.overall_confidence:.2f}")
    print(f"  Violations: {len(result2.violations)}")
    print(f"  Human Review Required: {result2.human_review_required}")
    
    print(f"\nSchema Violations:")
    for violation in result2.violations:
        print(f"  ❌ {violation.field_path}: {violation.message} ({violation.severity})")
    
    print(f"\nLow Confidence Fields:")
    for field_path in result2.low_confidence_fields:
        confidence = result2.field_confidences[field_path]
        reasons = ', '.join(confidence.reasons[:2])
        print(f"  🚨 {field_path}: {confidence.confidence_score:.2f} ({reasons})")

def test_resume_validation():
    """Test validation with resume schema"""
    print("\n\n🧪 Test 3: Resume Data Validation")
    print("-" * 35)
    
    validator = ValidationEngine(validation_level=ValidationLevel.LENIENT)
    
    # Load resume schema
    resume_schema = load_schema("testcases/convert your resume to this schema.json")
    
    # Sample extracted resume data
    resume_data = {
        "basics": {
            "name": "John Doe",
            "label": "Senior Software Engineer",
            "email": "john.doe@techcorp.com",
            "phone": "(555) 123-4567",
            "url": "https://johndoe.dev",
            "summary": "Experienced software engineer with 10+ years in web development",
            "location": {
                "city": "San Francisco",
                "countryCode": "US",
                "region": "CA"
            }
        },
        "work": [
            {
                "name": "Tech Corp",
                "position": "Senior Software Engineer",
                "startDate": "2020-01-01",
                "endDate": "2023-12-31",
                "summary": "Led development of scalable web applications",
                "highlights": [
                    "Increased performance by 40%",
                    "Led team of 5 developers"
                ]
            }
        ],
        "education": [
            {
                "institution": "UC Berkeley",
                "area": "Computer Science",
                "studyType": "Bachelor",
                "startDate": "2008-09-01",
                "endDate": "2012-06-01",
                "score": "3.8"
            }
        ],
        "skills": [
            {
                "name": "Programming",
                "keywords": ["JavaScript", "Python", "React", "Node.js"]
            }
        ]
    }
    
    source_text = """
    John Doe
    Senior Software Engineer
    Email: john.doe@techcorp.com
    Phone: (555) 123-4567
    Website: https://johndoe.dev
    
    Summary: Experienced software engineer with 10+ years in web development
    
    Experience:
    Tech Corp (2020-2023) - Senior Software Engineer
    Led development of scalable web applications
    - Increased performance by 40%
    - Led team of 5 developers
    
    Education:
    UC Berkeley (2008-2012) - Bachelor in Computer Science
    GPA: 3.8
    
    Skills: JavaScript, Python, React, Node.js
    """
    
    result = validator.validate_extraction(resume_data, resume_schema, source_text)
    
    print(f"Resume Validation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Overall Confidence: {result.overall_confidence:.2f}")
    print(f"  Total Fields Assessed: {len(result.field_confidences)}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Low Confidence Fields: {len(result.low_confidence_fields)}")
    
    # Show confidence distribution
    confidence_dist = {}
    for confidence in result.field_confidences.values():
        level = confidence.confidence_level.name
        confidence_dist[level] = confidence_dist.get(level, 0) + 1
    
    print(f"\nConfidence Distribution:")
    for level, count in confidence_dist.items():
        print(f"  {level}: {count} fields")
    
    # Show highest and lowest confidence fields
    sorted_confidences = sorted(
        result.field_confidences.items(),
        key=lambda x: x[1].confidence_score,
        reverse=True
    )
    
    print(f"\nHighest Confidence Fields:")
    for field_path, confidence in sorted_confidences[:5]:
        print(f"  ✅ {field_path}: {confidence.confidence_score:.2f}")
    
    print(f"\nLowest Confidence Fields:")
    for field_path, confidence in sorted_confidences[-5:]:
        reasons = ', '.join(confidence.reasons[:2])
        print(f"  ⚠️  {field_path}: {confidence.confidence_score:.2f} ({reasons})")

def test_github_actions_validation():
    """Test validation with GitHub Actions schema"""
    print("\n\n🧪 Test 4: GitHub Actions Validation")
    print("-" * 35)
    
    validator = ValidationEngine(validation_level=ValidationLevel.STRICT)
    
    # Load GitHub Actions schema
    github_schema = load_schema("testcases/github_actions_schema.json")
    
    # Sample extracted GitHub Actions data
    github_data = {
        "name": "MkDocs Publisher",
        "description": "A simple action to build an MkDocs site and push it to the gh-pages branch",
        "author": "DevRel Team",
        "inputs": {
            "python-version": {
                "description": "The version of Python to set up for building",
                "default": "3.11"
            },
            "requirements-file": {
                "description": "Path to the Python requirements file",
                "required": True
            },
            "gh-token": {
                "description": "GitHub token for deployment",
                "required": True
            }
        },
        "outputs": {
            "page-url": {
                "description": "The URL of the deployed GitHub Pages site"
            }
        },
        "runs": {
            "using": "composite",
            "steps": [
                {
                    "uses": "actions/checkout@v4",
                    "name": "Checkout Code"
                },
                {
                    "uses": "actions/setup-python@v5",
                    "name": "Setup Python",
                    "with": {
                        "python-version": "${{ inputs.python-version }}"
                    }
                }
            ]
        },
        "branding": {
            "color": "blue",
            "icon": "book-open"
        }
    }
    
    source_text = """
    MkDocs Publisher Action
    
    A simple action to build an MkDocs site and push it to the gh-pages branch.
    Author: DevRel Team
    
    Inputs:
    - python-version: Python version (default: 3.11)
    - requirements-file: Path to requirements file (required)
    - gh-token: GitHub token (required)
    
    Outputs:
    - page-url: Deployed site URL
    
    Uses composite action with steps for checkout and Python setup.
    Branding: blue color, book-open icon
    """
    
    result = validator.validate_extraction(github_data, github_schema, source_text)
    
    print(f"GitHub Actions Validation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Overall Confidence: {result.overall_confidence:.2f}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Low Confidence Fields: {len(result.low_confidence_fields)}")
    
    if result.violations:
        print(f"\nViolations:")
        for violation in result.violations[:5]:
            print(f"  ❌ {violation.field_path}: {violation.message}")
    
    if result.human_review_required:
        print(f"\n{validator.get_human_review_report(result)}")

def test_confidence_factors():
    """Test different confidence factors"""
    print("\n\n🧪 Test 5: Confidence Factor Analysis")
    print("-" * 40)
    
    validator = ValidationEngine()
    
    test_schema = {
        "type": "object",
        "required": ["name", "email"],
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
            "website": {"type": "string", "format": "url"},
            "status": {"type": "string", "enum": ["active", "inactive"]},
            "score": {"type": "integer", "minimum": 0, "maximum": 100}
        }
    }
    
    test_cases = [
        {
            "name": "High Confidence Case",
            "data": {
                "name": "Alice Smith",
                "email": "alice.smith@company.com",
                "website": "https://alice.dev",
                "status": "active",
                "score": 95
            },
            "source": "Alice Smith, email: alice.smith@company.com, website: https://alice.dev, status: active, score: 95"
        },
        {
            "name": "Medium Confidence Case",
            "data": {
                "name": "Bob",
                "email": "bob@email.com",
                "website": "http://bob.com",
                "status": "active",
                "score": 75
            },
            "source": "Bob has an email and website. His score is decent."
        },
        {
            "name": "Low Confidence Case",
            "data": {
                "name": "",
                "email": "invalid-email",
                "website": "not-a-url",
                "status": "unknown",
                "score": 150
            },
            "source": "Some person with unclear information."
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 {test_case['name']}:")
        result = validator.validate_extraction(
            test_case['data'], 
            test_schema, 
            test_case['source']
        )
        
        print(f"  Overall Confidence: {result.overall_confidence:.2f}")
        print(f"  Violations: {len(result.violations)}")
        
        # Show detailed confidence breakdown
        for field_path, confidence in result.field_confidences.items():
            print(f"    {field_path}: {confidence.confidence_score:.2f} - {', '.join(confidence.reasons[:2])}")

if __name__ == "__main__":
    test_validation_scenarios()
    test_resume_validation()
    test_github_actions_validation()
    test_confidence_factors() 