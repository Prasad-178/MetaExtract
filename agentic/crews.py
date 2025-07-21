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
        """Create an enhanced but streamlined extraction crew with 3 specialized agents"""
        
        # Create 3 focused agents
        schema_agent = self.agent_factory.create_schema_analyzer_agent()
        extraction_agent = self.agent_factory.create_extraction_specialist_agent()
        qa_agent = self.agent_factory.create_quality_assurance_agent()
        
        # Task 1: Schema Analysis (provides extraction guidance)
        schema_task = Task(
            description=f"""
            Analyze the JSON schema to provide extraction guidance for the team.
            
            SCHEMA TO ANALYZE:
            {json.dumps(schema, indent=1)[:1500]}...
            
            Your analysis should focus on:
            1. Key required fields that must be extracted
            2. Optional fields that would be valuable 
            3. Data types and structure requirements
            4. Any complex nested objects or arrays
            5. Validation rules or constraints
            
            Provide clear, actionable guidance for the extraction team:
            - What are the most important fields to extract?
            - What's the expected data structure?
            - Any special requirements or constraints?
            
            Keep your analysis practical and focused on helping extract data effectively.
            """,
            agent=schema_agent,
            expected_output="Clear extraction guidance based on schema analysis"
        )
        
        # Task 2: Data Extraction (the core extraction work)
        extraction_task = Task(
            description=f"""
            Extract data from the text document using the schema guidance from the previous task.

            TEXT DOCUMENT:
            {text[:3000]}

            EXTRACTION INSTRUCTIONS:
            1. Use the schema analysis guidance from the previous task
            2. Read the text carefully and extract actual values mentioned
            3. Follow the expected data structure and types
            4. Extract both required and optional fields when available
            5. Maintain proper JSON structure and formatting

            Example extractions from this text:
            - Look for "Action Name: MkDocs Publisher" → extract as "name": "MkDocs Publisher"
            - Look for "Author should be listed as 'DevRel Team'" → extract as "author": "DevRel Team"
            - Look for input descriptions → extract as properly structured input objects
            - Look for step descriptions → extract as properly structured steps

            Create a complete JSON object with all the information you can extract from the text.
            Focus on accuracy and completeness while following the schema structure.

            Return ONLY the extracted JSON object.
            """,
            agent=extraction_agent,
            expected_output="Complete JSON object with data extracted from the input text",
            context=[schema_task]  # Use schema analysis as context
        )
        
        # Task 3: Quality Assurance (validation and confidence scoring)
        qa_task = Task(
            description=f"""
            Review and validate the extracted data from the previous task.
            
            VALIDATION CHECKLIST:
            1. Check if the JSON is valid and well-formed
            2. Verify that required fields are present
            3. Validate data types match schema expectations
            4. Ensure extracted values make sense in context
            5. Check for any obvious errors or inconsistencies
            6. Assess completeness of the extraction
            
            SOURCE TEXT FOR VERIFICATION:
            {text[:2000]}...
            
            Your job:
            - Validate the extracted data against the schema
            - Fix any obvious errors or formatting issues
            - Ensure all values are properly extracted from the source text
            - Return the final, validated JSON object
            - If you find issues, correct them in the final output
            
            CRITICAL: Return ONLY the final, validated JSON object with any corrections applied.
            """,
            agent=qa_agent,
            expected_output="Final validated and corrected JSON object",
            context=[extraction_task]  # Use extraction result as context
        )
        
        return Crew(
            agents=[schema_agent, extraction_agent, qa_agent],
            tasks=[schema_task, extraction_task, qa_task],
            process=Process.sequential,
            verbose=False,
            full_output=True
        ) 