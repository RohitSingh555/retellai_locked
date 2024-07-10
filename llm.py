from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import json
from custom_types import CustomLlmRequest, CustomLlmResponse, Utterance
from typing import List
import pdfplumber
load_dotenv(override=True)

api_key = os.environ['OPENAI_API_KEY']

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_information_ai(text):
    
    openai = OpenAI(api_key=api_key)

    prompt = """
    Extract the personal_information, experience, education, and skills from the resume text below.
    Return a JSON response in the following output format:

    <Output format>
    {
    "personal_information": {
        "Name": "Extract the name of the candidate.",
        "Email": "Extract the email address of the candidate.",
        "Phone": "Extract the phone number of the candidate.",
        "Address": "Extract the address of the candidate.",
        "Date of Birth": "Extract the date of birth of the candidate.",
        "LinkedIn": "Extract the LinkedIn profile link of the candidate.",
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
        "what is the most creative thing you did in a work setting?"
        ]
    }
    </Output format>

    <Resume text>
    [Resume Text]
    </Resume text>
    """
    
    # You can use these to add a question of your choice
    # "questions": [
    #     "Can you tell me more about your experience at [Company] as a [Position]?",
    #     "How did your education at [College] contribute to your skills in [Degree]?",
    #     "What are some of the projects you worked on that showcase your skills in [skills]?",
    #     "Could you elaborate on any certifications you have obtained in [certifications]?",
    #     "What motivated you to pursue a career in [Position]? Tell me more about your introduction."
    # ]

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt.replace("[Resume Text]", text),
        max_tokens=1000,
        temperature=0.1,
        n=1,
        stop=["##"]
    )

    extracted_info = response.choices[0].text.strip()

    return extracted_info



# script_dir = os.path.dirname(os.path.realpath(__file__))
# pdf_file = os.path.join(script_dir, "kfengresume.pdf")

# Usage
pdf_file = "./resume_examples/Katherine Feng Resume 2023.pdf"
resume_text = extract_text_from_pdf(pdf_file)
print(resume_text)

# resume_text = extract_text_from_pdf(pdf_file)
# print(pdf_file)

extracted_info_json = extract_information_ai(resume_text)

try:
    extracted_info = json.loads(extracted_info_json)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    extracted_info = {}  # Set to an empty dictionary to allow further processing without breaking

Company_name = "Posh"
job_role = "Engineering Manager"
job_description="""
About POSH

We are all social creatures, but the dominant “social” companies today have evolved into digital loneliness machines, driving isolation, anxiety, and mental health challenges across our lives.

Human connection is lost. POSH is a beacon guiding us back.

POSH enables anyone to become an event organizer, building a community around their followers, and bringing people together in person to cultivate real-world human connections.

Founded by event enthusiasts and college dropouts, we’ve built the ultimate tools for creating, marketing, and monetizing in-person communities globally.

In just three years, POSH has grown to a team of 25, expanded to 2M users, secured $31M in venture funding, and facilitated over $90M in transactions.

About The Role

POSH has an extensive product, covering b2b and consumer social use cases, spanning a web and mobile application. You’ll be leading an engineering team that matches this diversity, consisting of highly motivated, high-achieving, full-stack SWEs at the junior, senior, and staff levels. You’ll be finding opportunities to drive technical excellence across all of POSH and aligning these initiatives with the career growth of our amazing team.

At a high level, you’ll be in charge of

Building a complex social and payments network with appropriate quality, security, and efficiency.

Optimizing our team, culture, processes, technology, and tools for high velocity at scale.

Directly leading our engineering team and driving their growth. Fostering a vibrant, performance and growth-oriented environment for our high-achiever team. 

Collaborating with partner teams, and C-suite to drive key POSH initiatives, influencing the experiences of our organizers and attendees.

Our ideal candidate

Has 4+ years of engineering management experience, with 7+ years of relevant software development industry experience in a fast-paced tech environment.

Has a diverse range of experience across a few companies, teams, and/or projects/domains.

Has experience with a high-traffic mobile application. Ideally has experience with consumer social or marketplace products.

Has good judgment in making tradeoffs to balance short-term business needs with long-term technical quality.

Has strong product and design instincts. Passionate about building delightfully designed experiences.

Has experience leading and shipping large initiatives with high business impact.

Has excellent communication and collaboration skills. Proven experience managing priorities between cross-functional partners.

Has extensive experience with the MERN stack, Typescript, and React Native, or similar technologies.

Creates a strong culture of operational excellence with a focus on raising the bar for quality, reliability, and availability.

Fosters an inclusive and engaging team environment.

Can identify, retain, grow, and acquire critical talent.

Compensation

Salary: $180,000-$220,000

Competitive Equity Package

Benefits:

100% Covered Health, Vision, & Dental Insurance

Equinox Membership

Unlimited PTO

Team Dinners, Offsites, and Events

New MacBook

Daily Uber Eats Credit

Relocation

This job is strictly in-person. Relocation bonuses are available for out-of-state candidates.
"""

try:
    # Access 'Name' and 'Position' safely
    person_name = extracted_info.get('personal_information', {}).get('Name', 'Applicant')
    first_name = person_name.split()[0] if person_name else 'Applicant'
    # job_position = extracted_info.get('experience', {}).get('Position', 'the position')

    begin_sentence = f"Hello {first_name}, Thanks for applying to {Company_name}. The purpose of this call is to help {Company_name} learn a little bit more about you to understand your fit for the {job_role} role as well as for you to learn about {Company_name}. We will leave time at the end for you to ask a few questions, so shall we start?"
except KeyError as e:
    print(f"Missing key: {e}")
    
print("Extracted Information:", extracted_info)
print(begin_sentence)

agent_prompt = """
<Objective>
You are a professional interviewer. You must always start the interview by confirming the pronunciation of the user's name and inquire about any nicknames they may have. 
Explicitly ask: 'Could you please tell me how to pronounce your name and if you have a nickname you prefer?' If a nickname is provided, address the user by their nickname for the
rest of the conversation. Then, proceed to assess the candidate's suitability for a specific role during a 5 to 10-minute call. Your primary objective is to gather comprehensive 
information about the candidate, including their fit for the role on multiple levels, their interests, problem-solving abilities, cultural fit, and other critical inputs for firms.
</Objective>

<Instructions>
Below is the conversation flow you should follow while conducting the interview:
1. Begin by saying, 'Before we proceed, may I confirm how to pronounce your name, and do you prefer a nickname?' Use the correct pronunciation and preferred name throughout the conversation.
2. After confirming the name, proceed with, 'Thank you, [Use Candidate’s Name]. Let me briefly describe what this role involves. You’ll be engaging in tasks that are crucial for [summarize key responsibilities and how they contribute to the company’s goals, creatively summarizing without quoting directly from the job description].'
3. Transition into the interview by asking, 'To get started, could you please share a little about your background and explain why you are interested in this role?' Avoid affirming the strength or quality of the candidate's background directly.
4. Use the candidate's responses, the job description, and the candidate's resume to constructively ask follow-up questions. Avoid repeating their words; instead, use their responses to dive deeper into more specific questions.
5. At the end of your questions, say, 'Do you have any questions for me about the role or the company?' Allow the candidate to ask 3-4 questions within the allotted time. If time is running out, politely inform the candidate by saying, 'We are almost out of time, so we can take one more question.'
6. Conclude the interview by thanking the candidate for their time and informing them of the next steps: 'Thank you for your time, [Candidate’s Name]. We will review your responses and get back to you soon. Have a great day!'

Below are key points to remember throughout the interview:
- Remember to maintain a neutral and professional dialogue, refraining from giving personal opinions, evaluative comments, or feedback.
- Incorporate creative and behavioral questions where appropriate. For example, 'Can you tell me about the most creative thing you've done in a work setting?'
- If the candidate does not respond, gently prompt with, 'Can you hear me okay, [Use Candidate’s Name], or would you like a moment?'
- Address inappropriate language or behavior by stating, 'It’s important for us to keep this conversation professional. If this continues, I may need to end the call early.'
- Respond to technical issues by asking, 'It seems there might be a connection issue. Can you hear me now?' Ensure the issue is resolved to continue effectively.
- Maintain a consistent and professional tone throughout the call to ensure clarity and convey respect.
- Your role as an interviewer encompasses various responsibilities, all centered around the candidate. 
- Your objective is to create a positive and productive atmosphere with job applicants, evaluating their qualifications and suitability for the position. This involves conducting thorough assessments of their skills and experiences, delving into their past work and achievements. 
- Engage in active listening and thoughtful questioning to gain insights into their capabilities and potential contributions to the organization. 
- Adhere to all hiring protocols and maintain confidentiality throughout the selection process. Your ultimate goal is to ensure a fair and effective recruitment process that aligns with the company's goals and values. 
- Communicate concisely and professionally, using clear and straightforward language while asking only questions. This approach fosters meaningful interactions and cultivates a positive candidate experience. 
- Your demeanor should be professional yet personable, demonstrating respect and empathy towards candidates while remaining objective in your assessments. 
- Strive to ask only questions and provide guidance only when necessary. 
- Keep your feedback concise and stay on topic during the conversation. Avoid awkward pauses and respond promptly. 
- Once you have gathered sufficient information about the user, conclude the call with a compliment and end the conversation. Don't stretch the conversation for more than 10 minutes.
</Instructions>
"""

agent_prompt = agent_prompt.format(extracted_info, "{}")

print(agent_prompt)
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
            "content": f'##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and do not reveal to the user if his responses were good or bad keep the results a secret, get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "pardon", "Hey, I can’t hear you; do you mind repeating yourself?" and be interactive in case of technical difficulties). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n. Do not speak while you are getting a response from the user and do not reply with "hmm" or "uhhuh" or anything that is unprofessional.\n Recognise If the answer of the user is bad(for ex: If I answer I write to a question can you explain your hobbies? then consider it as a bad answer as I did not explain why) or there was no answer at all then ask the question again a one more time to get a better response then move forward with the other questions.\n If the user is not saying anything for 5 seconds, Give Warnings(Sorry i can not hear any response from your side. i would have to end the call if this persists) to the user before you hang up in 5 seconds.\n never get aggressive, be polite throughout the call.\n In case of swearing + racism give warning(Hey {first_name}, unfortunately we don’t tolerate inappropriate comments or remarks. If you can not maintain a professional environment, I will have to end the call.), and if the behavior persists: end the call after greetings.\n Before ending ask a question to the user (if he/she has any questions for you(the AI agent) Answer based on {job_description} - never make up any info.\n ' + agent_prompt
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

