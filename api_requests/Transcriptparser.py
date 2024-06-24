import os
import json
from docx import Document
import openai
from openai import OpenAI

client = OpenAI()

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

resume_path = '../transcripts/6_resume.docx'
transcript_path = '../transcripts/6_transcript.docx'

resume_content = read_docx(resume_path)
transcript_content = read_docx(transcript_path)

job_description = "Senior Data Scientist at Moderna. This role involves developing and overseeing Immuno-Assay development, essential for Moderna's research and vaccine production efforts."
company_background = "Moderna is a biotechnology company pioneering messenger RNA (mRNA) therapeutics and vaccines. We aim to transform how medicines are created and delivered, focusing on preventing and fighting diseases."

prompt = f'''
<Instructions>
Your job is to assess a candidate based on their transcript and resume. Your goal is to determine whether this candidate is a good fit for the Senior Data Scientist role at Moderna. Answer the questions below according to the candidate’s transcript and the resume. Use the job description and company background to guide your responses. Return a JSON response and only a JSON response. 
</Instructions>

<Questions>
Provide a brief summary of the candidate.
What are the strengths of the candidate?
What are the weaknesses of the candidate?
Describe the cultural fit of the candidate. In other words, how does the candidate’s skills and ideals align with the company’s values?
Use all the information provided about the candidate to explain why the candidate should be accepted or rejected from the role.
</Questions>

<Job Description>
{job_description}
</Job Description>

<Company Background>
{company_background}
</Company Background>

<Resume>
{resume_content}
</Resume>

<Transcript>
{transcript_content}
</Transcript>
'''

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an AI assistant trained to evaluate job candidates based on their resume and interview transcript."},
        {"role": "user", "content": prompt}
    ]
)

response_content = response.choices[0].message
content_part = response_content.content
print(content_part)

with open('parsed_transcript5.json', 'w') as f:
    f.write(content_part)

print("Response saved to parsed_transcript5.json")
