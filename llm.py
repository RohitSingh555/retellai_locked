from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import json
from custom_types import CustomLlmRequest, CustomLlmResponse, Utterance
from typing import List
load_dotenv(override=True)

api_key = os.environ['OPENAI_API_KEY']

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_information_ai(text):
    # api_key = "sk-proj-eZkmifxveCKsTRQT4KSfT3BlbkFJxhbZLPLS55Ai3cHSyEVC"
    openai = OpenAI(api_key=api_key)

    prompt = """
    Extract the following information from the provided resume text:

    [Resume Text]

    Please provide the extracted information in JSON format with the following structure:

    {
        "personal_information": {
            "Name": "Extract the name of the candidate.",
            "Email": "Extract the email address of the candidate.",
            "Phone": "Extract the phone number of the candidate.",
            "Address": "Extract the address of the candidate.",
            "Date of Birth": "Extract the date of birth of the candidate.",
            "LinkedIn": "Extract the LinkedIn profile link of the candidate."
            "introduction": "Extract the string where user tells about himself/explains what he has done and not done."
        },
        "experience": {
            "Position": "Extract the position held by the candidate.",
            "Company": "Extract the company name where the candidate worked.",
            "Start Date": "Extract the start date of employment.",
            "End Date": "Extract the end date of employment."
        },
        "education": {
            "Degree": "Extract the degree obtained by the candidate.",
            "College": "Extract the college or university name.",
            "Start Date": "Extract the start date of education.",
            "End Date": "Extract the end date of education."
        },
        "skills": "Extract the skills mentioned by the candidate.",
        "projects": "Extract the project names along with relevant information.",
        "certifications": "Extract any certifications obtained by the candidate.",

        "questions": [
            "Can you tell me more about your experience at [Company] as a [Position]?",
            "How did your education at [College] contribute to your skills in [Degree]?",
            "What are some of the projects you worked on that showcase your skills in [skills]?",
            "Could you elaborate on any certifications you have obtained in [certifications]?",
            "What motivated you to pursue a career in [Position]? Tell me more about your introduction."
        ]
    }
    """

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt.replace("[Resume Text]", text),
        max_tokens=1000,
        n=1,
        stop=["##"]
    )

    extracted_info = response.choices[0].text.strip()

    return extracted_info



script_dir = os.path.dirname(os.path.realpath(__file__))
pdf_file = os.path.join(script_dir, "resume.pdf")

resume_text = extract_text_from_pdf(pdf_file)

extracted_info = extract_information_ai(resume_text)
# extracted_info = {
#         "personal_information": {
#             "Name": "Sanket Sarwade",
#             "Email": "sanketsarwade111@gmail.com",
#             "Phone": "7798248452",
#             "Address": "New Cidco, Nashik",
#             "Date of Birth": "Jul 15, 2001",
#             "LinkedIn": "https://github.com/sanketsarwade",
#             "introduction": "As a highly motivated and detail-oriented data scientist, I am eager to begin my career in the eld of data science. With a solid foundation in statistics, programming, and machine learning techniques, I am well-equipped to tackle complex data problems and deliver meaningful insights. Through my academic and personal projects, I have honed my skills in data analysis, visualization, and modeling. I am procient in using tools such as Python, SQL, and Tableau, and have experience working with various data sources." 
#         },
#         "experience": {
#             "Position": "Data Scientist",
#             "Company": "Data Camp",
#             "Start Date": "Oct 2022",
#             "End Date": "Jan 2023"
#         },
#         "education": {
#             "Degree": "Bsc Microbiology",
#             "College": "Sinhgad College of Science",
#             "Start Date": "Jun 2019",
#             "End Date": "Sep 2022"
#         },
#         "skills": ["Python", "SQL", "Machine Learning", "Deep Learning", "Tableau", "Data Visualization", "Data Management", "Leadership", "Handling Pressure", "Collaboration"],     
#         "projects": [
#             {"name": "Human Activity Recognition", "description": "Worked on a project using Kaggle datasets to classify different human activities", "tools_used": ["Python", "Machine Learning"]},
#             {"name": "Email Spam", "description": "Built a model to identify and filter out spam emails using machine learning algorithms", "tools_used": ["Python", "SQL"]},
#             {"name": "Breast Cancer Prediction", "description": "Utilized data from a study to accurately predict the presence of breast cancer in patients", "tools_used": ["Python", "Tableau"]},
#             {"name": "Anomaly Detection", "description": "Developed a system to identify and flag anomalous data points in a dataset", "tools_used": ["Python", "Machine Learning"]}, 
#             {"name": "IPL Prediction", "description": "Collaborated with a team to predict the outcomes of IPL matches using historical data", "tools_used": ["Python", "Data Visualization"]}
#         ],

#         "certifications": ["Python", "Machine Learning", "Deep Learning", "SQL", "Data Science", "Tableau"],
#         "questions": [
#             "Can you tell me more about your experience at Data Camp as a Data Scientist?",
#             "How did your education at Sinhgad College of Science contribute to your skills in Bsc Microbiology?",
#             "What are some of the projects you worked on that showcase your skills in Python, SQL, and Machine Learning?",
#             "Could you elaborate on any certifications you have obtained in Python, Machine Learning, and SQL?",
#             "What motivated you to pursue a career in Data Science? Tell me more about your introduction."
#         ]
#     }

# print(extracted_info)

# Convert extracted information to JSON format
# extracted_info_json = json.loads(extracted_info)
job_role="Fullstack Developer"
Company_name="Agilemorph Solutions"
job_desc="We are looking for a passionate Python developer to join our team at Agilemorph Solutions.You will be responsible for developing and implementing high-quality software solutions, creating complex applications using cutting-edge programming features and frameworks and collaborating with other teams in the firm to define, design and ship new features.As an active part of our company, you will brainstorm and chalk out solutions to suit our requirements and meet our business goals. You will also be working on data engineering problems and building data pipelines. You would get ample opportunities to work on challenging and innovative projects, using the latest technologies and tools. If you enjoy working in a fast-paced and collaborative environment, we encourage you to apply for this exciting role. We offer industry-standard compensation packages, relocation assistance, and professional growth and development opportunities."

print("Extracted Information:",extracted_info)
begin_sentence = f"Hello {extracted_info['personal_information']['Name']}, Welcome to the {Company_name} interview screening. I am here to assess your suitability for the {extracted_info['experience']['Position']} position."

agent_prompt = f"As a professional interviewer call the callee by his/her name always, then first: Ask 5 questions from here: {extracted_info['questions']}, your responsibilities are multifaceted and candidate-centered. You aim to establish a positive and productive atmosphere with job applicants, evaluating their qualifications and fit for the position. Your role involves conducting thorough assessments of candidate skills and experiences, probing into their past work and achievements. Engage in active listening and thoughtful questioning to gain insights into their capabilities and potential contributions to the organization. Regular communication and feedback with candidates are essential for guiding them through the hiring process and providing clarity on expectations. Additionally, you adhere to all hiring protocols and maintain confidentiality throughout the selection process. Your goal is to ensure a fair and effective recruitment process that aligns with the company's goals and values. Communicate concisely and professionally. Aim for responses in clear and straightforward language, keeping exchanges focused and purposeful. This approach facilitates meaningful interactions and fosters a positive candidate experience. Your approach should be professional yet personable, demonstrating respect and empathy towards candidates while maintaining objectivity in your assessments. Strive to build rapport and trust with applicants, encouraging open communication and honest dialogue. It's important to provide constructive feedback and guidance, helping candidates understand areas for improvement and maximizing their potential for success. .".format(extracted_info, "{}")

class LlmClient:
    def __init__(self):
        self.client = OpenAI(
            # organization=os.environ['OPENAI_ORGANIZATION_ID'],
            api_key=os.environ['OPENAI_API_KEY'],
        )
    
    def draft_begin_message(self):
        response = CustomLlmResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response
    
    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance["role"] == "agent":
                messages.append({
                    "role": "assistant",
                    "content": utterance['content']
                })
            else:
                messages.append({
                    "role": "user",
                    "content": utterance['content']
                })
        return messages

    def prepare_prompt(self, request: CustomLlmRequest):
        prompt = [{
            "role": "system",
            "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n' +
          agent_prompt
        }]
        transcript_messages = self.convert_transcript_to_openai_messages(request.transcript)
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append({
                "role": "user",
                "content": "Hey! You there?",
            })
        return prompt

    def draft_response(self, request: CustomLlmRequest):      
        prompt = self.prepare_prompt(request)
        stream = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=prompt,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response = CustomLlmResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response
        
        response = CustomLlmResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response
