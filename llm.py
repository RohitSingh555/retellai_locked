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
pdf_file = os.path.join(script_dir, "ssresume.pdf")

resume_text = extract_text_from_pdf(pdf_file)

extracted_info_json = extract_information_ai(resume_text)

try:
    extracted_info = json.loads(extracted_info_json)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    extracted_info = {}  # Set to an empty dictionary to allow further processing without breaking

Company_name = "PermitFlow"
job_role = "distinguished Head of Strategy, a pivotal leadership role designed to serve as the 'second CEO.'"
job_description="""About PermitFlow
PermitFlow's mission is to streamline and simplify construction permitting in $1.6 trillion United States construction market. Our software reduces time to permit, supporting permitting end-to-end including permit research, application preparation, submission, and monitoring.

We’ve raised a $31m Series A led by Kleiner Perkins with participation from Initialized Capital, Y Combinator, Felicis Ventures, Altos Ventures, and the founders and executives from Zillow, PlanGrid, Thumbtack, Bluebeam, Uber, Procore, and more.

Our team is remote first and consists of architects, structural engineers, permitting experts, and workflow software specialists, all who have personally experienced the pain of permitting.

What You’ll Do:
At PermitFlow, we are in pursuit of a distinguished Head of Strategy, a pivotal leadership role designed to serve as the "second CEO." This individual will play an instrumental role in guiding our B2B construction tech startup through its next phase of growth. As a beacon of strategic direction, you will work hand-in-glove with the Co-Founders, influencing major decisions and propelling the organization towards achieving its ambitious goals. This role is not for the faint-hearted; it demands a combination of rigorous analytical prowess, strategic foresight, and the agility to lead high-impact initiatives across the board.

Your role will primarily focus on:

Strategic Influence: Act as the primary strategic advisor to the Co-Founders, offering critical insights and guidance on all aspects of the business. Your role will encompass a wide range of responsibilities, from shaping the company's revenue and GTM strategy to steering major roadmap investments and leading investor relations and fundraising efforts.

Analytical Leadership: Employ a data-driven approach to problem-solving, leveraging your exceptional quantitative analysis skills to dissect complex challenges and drive informed decision-making. You will be expected to structure and execute strategic projects independently, demonstrating creativity and a keen eye for detail.

Collaborative Excellence: Forge strong alliances across the organization, working closely with the Head of People on talent and rewards strategies, and partnering with Product on roadmap prioritization. Your ability to communicate effectively and stand in for the CEO when necessary will be crucial in maintaining alignment and fostering a culture of collaboration.

Operational Agility: Oversee a broad spectrum of operational and strategic tasks, ensuring the seamless execution of company objectives. Your proactive mindset will be key in anticipating needs and navigating the dynamic landscape of a hypergrowth startup.

The ideal candidate for this role is a proactive team player, strategic thinker and able to use discretion when necessary.

Qualifications & Fit:
A minimum of 2 years' experience in a consulting role at a top-tier firm (e.g., Bain & Co.), or a similar position in a high-growth technology environment, with a strong preference for backgrounds in real estate or construction.

Demonstrated expertise in quantitative analysis and research, with proven ability to deliver insights and strategies that drive business outcomes.

Exceptional communication and presentation skills, capable of articulating complex ideas clearly and compellingly, both internally and externally.

A track record of leading strategic projects from inception to success, showcasing an ability to think creatively and act decisively.

An entrepreneurial spirit, underscored by an ownership mentality and the capability to thrive in ambiguous situations with shifting priorities.

Bonus points for entrepreneurial experience, ownership mentality, and experience at an early stage startup.

Benefits
Competitive salary and equity packages

Home office & equipment stipend

Flexible working hours & unlimited PTO

Health, dental, and vision insurance"""

try:
    # Access 'Name' and 'Position' safely
    person_name = extracted_info.get('personal_information', {}).get('Name', 'Applicant')
    first_name = person_name.split()[0] if person_name else 'Applicant'
    job_position = extracted_info.get('experience', {}).get('Position', 'the position')

    begin_sentence = f"Hello {first_name}, Thanks for applying to {Company_name}. We'd like to know you more before we proceed with assessing your suitability for the role of a {job_role}."
except KeyError as e:
    print(f"Missing key: {e}")
    begin_sentence = "Welcome to the interview screening."

print("Extracted Information:", extracted_info)
print(begin_sentence)

agent_prompt =  f"As a professional interviewer call the user by his/her first name always, then firstly: ask him/her a few generic questions like (tell me about yourself then after the response ask a few questions based on the response. ) before you dive into the interviews after that Ask 5 questions from here: {extracted_info['questions']}, your responsibilities are multifaceted and candidate-centered. You aim to establish a positive and productive atmosphere with job applicants, evaluating their qualifications and fit for the position. Your role involves conducting thorough assessments of candidate skills and experiences, probing into their past work and achievements. Engage in active listening and thoughtful questioning to gain insights into their capabilities and potential contributions to the organization. Adhere to all hiring protocols and maintain confidentiality throughout the selection process. Your goal is to ensure a fair and effective recruitment process that aligns with the company's goals and values. Communicate concisely and professionally. Aim for responses in clear and straightforward language, Asking only questions. This approach facilitates meaningful interactions and fosters a positive candidate experience. Your approach should be professional yet personable, demonstrating respect and empathy towards candidates while maintaining objectivity in your assessments. Strive to ask Only question and respond to answers in two to three words only. It's important to provide guidance only when needed, Be as concise as possible with the feedback and make sure you are on topic everytime you speak. Don't take awkward pauses be quick at responding.".format(extracted_info, "{}")
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
            "content": f'##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "pardon", "can you hear me? and be interactive in case of technical difficulties"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n. Do not speak while you are getting a response from the user and do not reply with "hmm" or "uhhuh" or anything that is unprofessional.\n Recognise If the answer of the user is bad(for ex: If I answer I write to a question can you explain your hobbies? then consider it as a bad answer as I did not explain why) or there was no answer at all then ask the question again a one more time to get a better response then move forward with the other questions.\n If the user is not saying anything for 5 seconds, Give Warnings(Sorry i can not hear any response from your side. i would have to end the call if this persists) to the user before you hang up in 5 seconds.\n never get aggressive, be polite throughout the call.\n In case of swearing + racism give warning(Sorry we do not entertain such remarks. i would have to end the call if this persists), and if the behavior persists: end the call after greetings.\n Before ending ask a question to the user (if he/she has any questions for you(the AI agent) Answer based on {job_description} - never make up any info.\n' +
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
