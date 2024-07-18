import os
import json
from docx import Document
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from dotenv import load_dotenv
load_dotenv(override=True)

openai.api_key = os.getenv("OPENAI_ID")
# client = OpenAI()

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

class ResumeAnalysis(BaseModel):
    summary: str = Field(description='return a string output that summarizes the candidate')
    strengths: str = Field(description='return a string output that answers the question in detail')
    weaknesses: str = Field(description='return a string output that answers the question in detail')
    cultural_fit: str = Field(description='return a string output that answers the question in detail')
    acceptance: str = Field(description='return a string output that answers the question in detail')

def setup_model():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    human_message = '''
    <Role>
    You are an AI assistant trained to evaluate job candidates based on their resume and interview transcript.
    <Role>
    
    <Instructions>
    Your job is to assess a candidate based on their transcript and resume. 
    Your goal is to determine whether this candidate is a good fit for the Senior Data Scientist role at Moderna. 
    Answer the questions below according to the candidate’s transcript and the resume and return a JSON response.
    Use the job description and company background to guide your responses. Return a JSON response and only a JSON response. 
    The JSON response should include the following keys: summary, strengths, weaknesses, cultural_fit, acceptance.
    </Instructions>

    <Questions>
    1. Provide a brief summary of the candidate.
    2. What are the strengths of the candidate?
    3. What are the weaknesses of the candidate?
    4. Describe the cultural fit of the candidate. In other words, how does the candidate’s skills and ideals align with the company’s values?
    5. Use all the information provided about the candidate to explain why the candidate should be accepted or rejected from the role.
    </Questions>

    <ResponseFormat>
    {format_instructions}
    </ResponseFormat>

    <Job Description>
    {job_description}
    </Job Description>

    <Company Background>
    {company_background}
    </Company Background>

    <Content>
    {content}
    </Content>

    Json Response:

    '''

    parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

    prompt = PromptTemplate(
        template = human_message,
        input_variables = ["job_description", "company_background", "content"],
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    return chain


def process_files_in_folder(folder_path, output_folder, job_description, company_background):
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_id = filename.split('_')[0]
            output_filename = f'{file_id}_parsed_transcript.json'
            output_filepath = os.path.join(output_folder, output_filename)
            
            if os.path.exists(output_filepath):
                print(f"Skipping {filename} as {output_filename} already exists.")
                continue
            
            file_path = os.path.join(folder_path, filename)
            content = read_docx(file_path)

            chain = setup_model()
            response = chain.invoke({"job_description": job_description, "company_background": company_background, "content": content})
            response_parsed = json.loads(response.json())

            with open(output_filepath, 'w') as f:
                json.dump(response_parsed, f, indent=4)
            print(response_parsed)

            print(f"Response saved to {output_filename}")


job_description = "Senior Data Scientist at Moderna. This role involves developing and overseeing Immuno-Assay development, essential for Moderna's research and vaccine production efforts."
company_background = "Moderna is a biotechnology company pioneering messenger RNA (mRNA) therapeutics and vaccines. We aim to transform how medicines are created and delivered, focusing on preventing and fighting diseases."

transcripts_folder = '../transcripts'
output_folder = transcripts_folder  # Output in the same folder

process_files_in_folder(transcripts_folder, output_folder, job_description, company_background)
