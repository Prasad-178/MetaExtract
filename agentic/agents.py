"""
Simplified agents for agentic extraction approach - optimized for speed and cost
"""

from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import List, Optional
import os


def create_llm(temperature: float = 0.1, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Create OpenAI LLM instance with cheaper model"""
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


class AgentFactory:
    """Simplified factory for creating core extraction agents"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def create_schema_analyzer_agent(self) -> Agent:
        """Create agent specialized in schema analysis"""
        return Agent(
            role='Schema Analysis Expert',
            goal='Quickly analyze JSON schemas to understand structure and extraction requirements',
            backstory="""You are an expert in JSON schema analysis. You quickly understand 
            data structures and identify key fields that need to be extracted. You focus on 
            practical extraction guidance rather than complex analysis.""",
            verbose=False,
            allow_delegation=False,
            llm=create_llm(temperature=0.0),
            tools=[]
        )
    
    def create_extraction_specialist_agent(self) -> Agent:
        """Create agent specialized in data extraction"""
        return Agent(
            role='Data Extraction Expert',
            goal='Extract structured data from text with high accuracy and speed',
            backstory="""You are a master at extracting structured information from unstructured text.
            You have exceptional attention to detail and can quickly identify relevant data points. 
            You always return clean, valid JSON that matches the required schema exactly.""",
            verbose=False,
            allow_delegation=False,
            llm=create_llm(temperature=0.1),
            tools=[]
        )
    
    def create_quality_assurance_agent(self) -> Agent:
        """Create agent specialized in quality assurance and validation"""
        return Agent(
            role='Quality Assurance Expert',
            goal='Validate and refine extracted data to ensure highest quality',
            backstory="""You are a quality assurance expert who reviews extracted data for 
            accuracy and completeness. You ensure the data matches the schema perfectly and 
            fix any issues. You always output the final, validated JSON.""",
            verbose=False,
            allow_delegation=False,
            llm=create_llm(temperature=0.0),
            tools=[]
        )
    
    def create_simple_extraction_team(self) -> List[Agent]:
        """Create simplified team of 3 core agents"""
        return [
            self.create_schema_analyzer_agent(),
            self.create_extraction_specialist_agent(),
            self.create_quality_assurance_agent()
        ] 