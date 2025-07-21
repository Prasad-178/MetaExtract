"""
Simplified crew workflows for fast agentic extraction
"""

from crewai import Crew, Task, Process
from typing import Dict, Any, List, Optional
import json

from .agents import AgentFactory


class SimplifiedTaskFactory:
    """Simplified factory for creating extraction tasks"""
    
    @staticmethod
    def create_schema_analysis_task(schema: Dict[str, Any], agent) -> Task:
        """Create simplified schema analysis task"""
        schema_str = json.dumps(schema, indent=1)[:1000]  # Limit schema size
        return Task(
            description=f"""
            Analyze this JSON schema and provide guidance for data extraction:
            
            Schema: {schema_str}
            
            Provide:
            1. Key fields to extract
            2. Data types expected
            3. Any special requirements
            
            Keep your analysis brief and focused on extraction guidance.
            """,
            agent=agent,
            expected_output="Brief analysis focusing on key extraction requirements"
        )
    
    @staticmethod
    def create_extraction_task(text: str, schema: Dict[str, Any], agent) -> Task:
        """Create core extraction task"""
        # Limit text size for faster processing
        text_preview = text[:4000] + ("..." if len(text) > 4000 else "")
        
        return Task(
            description=f"""
            IMPORTANT: Extract ACTUAL VALUES from the text below. Do NOT create schema definitions.

            TEXT TO READ AND EXTRACT FROM:
            {text_preview}

            Your job: Read the text above and find these actual values mentioned in it:
            
            Example of what you should extract:
            - If the text says "Action Name: MkDocs Publisher" → extract "MkDocs Publisher" as the name
            - If the text says "Author should be listed as 'DevRel Team'" → extract "DevRel Team" as the author  
            - If the text says "Purpose: A simple action to build..." → extract that as the description
            
            Create a JSON object with the ACTUAL VALUES you find:
            {{
                "name": "actual name found in text",
                "description": "actual description found in text", 
                "author": "actual author found in text",
                "inputs": {{"actual input names": {{"description": "actual descriptions"}}}},
                "outputs": {{"actual output names": {{"description": "actual descriptions"}}}},
                "runs": {{"using": "execution type", "steps": ["actual steps"]}},
                "branding": {{"color": "actual color", "icon": "actual icon"}}
            }}
            
            CRITICAL: Extract the REAL VALUES from the text, not schema templates.
            
            Return ONLY the JSON with actual extracted values.
            """,
            agent=agent,
            expected_output="JSON object with actual values extracted from the input text"
        )
    
    @staticmethod
    def create_validation_task(agent) -> Task:
        """Create validation and refinement task"""
        return Task(
            description=f"""
            Review the extracted data from the previous task and ensure it's perfect.
            
            Your job:
            1. Validate the JSON structure matches the schema
            2. Check for any missing critical information
            3. Fix any formatting issues
            4. Return the final, polished JSON
            
            CRITICAL: Return ONLY the final valid JSON object, nothing else.
            """,
            agent=agent,
            expected_output="Final validated JSON object"
        )


class SimplifiedCrewFactory:
    """Simplified factory for creating extraction crews"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.agent_factory = AgentFactory(api_key)
        self.task_factory = SimplifiedTaskFactory()
    
    def create_fast_extraction_crew(self, text: str, schema: Dict[str, Any]) -> Crew:
        """Create a fast, simplified extraction crew"""
        
        # Create just 1 focused extraction agent to avoid confusion
        extraction_agent = self.agent_factory.create_extraction_specialist_agent()
        
        # Create a single, very clear extraction task
        extraction_task = Task(
            description=f"""
            You are extracting data from a text document. Read the text below and extract the actual information mentioned in it.

            TEXT DOCUMENT:
            {text[:3000]}

            Extract the following information from the text above (only what is actually mentioned):

            1. NAME: Look for phrases like "Action Name:" or "name:" - extract the actual name
            2. DESCRIPTION: Look for "Purpose:" or "description" - extract the actual description  
            3. AUTHOR: Look for "Author" mentions - extract the actual author name
            4. INPUTS: Look for "Inputs Needed:" section - extract actual input names and descriptions
            5. OUTPUTS: Look for "Outputs:" section - extract actual output names and descriptions
            6. STEPS: Look for "steps" or execution workflow - extract actual steps mentioned
            7. BRANDING: Look for color and icon mentions - extract actual values

            Example of what to extract from text:
            - Text: "Action Name: MkDocs Publisher" → Extract: "MkDocs Publisher"
            - Text: "Author should be listed as 'DevRel Team'" → Extract: "DevRel Team"
            - Text: "color blue and the book-open icon" → Extract: color: "blue", icon: "book-open"

            Return a JSON object with the actual values you found:
            {{
                "name": "actual name from text",
                "description": "actual description from text",
                "author": "actual author from text", 
                "inputs": {{
                    "actual-input-name": {{
                        "description": "actual description from text",
                        "required": true_or_false
                    }}
                }},
                "outputs": {{
                    "actual-output-name": {{
                        "description": "actual description from text"
                    }}
                }},
                "runs": {{
                    "using": "composite",
                    "steps": ["actual steps from text"]
                }},
                "branding": {{
                    "color": "actual color from text",
                    "icon": "actual icon from text"
                }}
            }}

            CRITICAL: Extract REAL VALUES from the text, not schema examples.
            """,
            agent=extraction_agent,
            expected_output="JSON object with actual data values extracted from the input text"
        )
        
        return Crew(
            agents=[extraction_agent],
            tasks=[extraction_task],
            process=Process.sequential,
            verbose=False,
            full_output=True
        ) 