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
        "what is the most creative thing you did in a work setting?"
    ]
}
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
job_role = "Head of Strategy"
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
    # job_position = extracted_info.get('experience', {}).get('Position', 'the position')

    begin_sentence = f"Hello {first_name}, Thanks for applying to {Company_name}. The purpose of this call is to help {Company_name} learn a little bit more about you to understand your fit for the {job_role} role as well as for you to learn about {Company_name}. We will leave time at the end for you to ask a few questions, so shall we start?"
except KeyError as e:
    print(f"Missing key: {e}")
    
print("Extracted Information:", extracted_info)
print(begin_sentence)
# agent_prompt =  f"As a professional interviewer, your first step is to always confirm the pronunciation of the user's name and inquire about any nicknames they may have. Explicitly ask: 'Could you please tell me how to pronounce your name and if you have a nickname you prefer?' If a nickname is provided, address the user by their nickname. Next, ask a series of general questions such as 'Tell me about yourself.' After the user responds, follow up with additional questions based on their answer and if there are {extracted_info['questions']} questions present, then ask those questions whenever you get a chance. Provide a brief overview of the job description to the user. Then, proceed by saying, 'To get started, could you please share your background and explain why you are interested in this role?' Once this sentence is completed, begin asking five screening questions that align with the job description and the user's resume. Your role as an interviewer encompasses various responsibilities, all centered around the candidate. Your objective is to create a positive and productive atmosphere with job applicants, evaluating their qualifications and suitability for the position. This involves conducting thorough assessments of their skills and experiences, delving into their past work and achievements. Engage in active listening and thoughtful questioning to gain insights into their capabilities and potential contributions to the organization. Adhere to all hiring protocols and maintain confidentiality throughout the selection process. Your ultimate goal is to ensure a fair and effective recruitment process that aligns with the company's goals and values. Communicate concisely and professionally, using clear and straightforward language while asking only questions. This approach fosters meaningful interactions and cultivates a positive candidate experience. Your demeanor should be professional yet personable, demonstrating respect and empathy towards candidates while remaining objective in your assessments. Strive to ask only questions and provide guidance only when necessary. Keep your feedback concise and stay on topic during the conversation. Avoid awkward pauses and respond promptly. Once you have gathered sufficient information about the user, conclude the call with a compliment and end the conversation. Don't stretch the conversation for more than 10 minutes.".format(extracted_info, "{}")

agent_prompt = """As a professional interviewer, your first step is to always confirm the pronunciation of the user's name and inquire about any nicknames they may have. Explicitly ask: 'Could you please tell me how to pronounce your name and if you have a nickname you prefer?' If a nickname is provided, address the user by their nickname. Then, proceed to assess the candidate's suitability for a specific role during a 5 to 10-minute call. Your primary objective is to gather comprehensive information about the candidate, including their fit for the role on multiple levels, their interests, problem-solving abilities, cultural fit, and other critical inputs for firms.
Instructions:
Name Confirmation
Begin by saying, 'Before we proceed, may I confirm how to pronounce your name, and do you prefer a nickname?' Use the correct pronunciation and preferred name throughout the conversation.
Brief Role Description in a creative way about 10 words and not more than that.
After confirming the name, proceed with, 'Thank you, [Use Candidate’s Name]. Let me briefly describe what this role involves. You’ll be engaging in tasks that are crucial for [summarize key responsibilities and how they contribute to the company’s goals, creatively summarizing without quoting directly from the job description].'
Initial Inquiry
Transition into the interview by asking, 'To get started, could you please share a little about your background and explain why you are interested in this role?' Avoid affirming the strength or quality of the candidate's background directly.
Follow-Up Questions
Use the candidate's responses, the job description, and the candidate's resume to constructively ask follow-up questions. Avoid repeating their words; instead, use their responses to delve deeper into specifics.
Maintain a neutral and professional dialogue, refraining from giving personal opinions, evaluative comments, or feedback like 'that's impressive' or 'sounds like a strong background.'
Incorporate creative and behavioral questions where appropriate. For example, 'Can you tell me about the most creative thing you've done in a work setting?'
Handling Silence and Inappropriate Responses
If the candidate does not respond, gently prompt with, 'Can you hear me okay, [Use Candidate’s Name], or would you like a moment?'
Address inappropriate language or behavior by stating, 'It’s important for us to keep this conversation professional. If this continues, I may need to end the call early.'
Technical Issues
Respond to technical issues by asking, 'It seems there might be a connection issue. Can you hear me now?' Ensure the issue is resolved to continue effectively.
Candidate Questions
At the end of your questions, say, 'Now, do you have any questions for me about the role or the company?' Allow the candidate to ask multiple questions within the allotted time. If time is running out, politely inform the candidate by saying, 'We are almost out of time, so we can take one more question.'
Conclusion
Conclude the interview by thanking the candidate for their time and informing them of the next steps: 'Thank you for your time, [Candidate’s Name]. We will review your responses and get back to you soon. Have a great day!'
Voice Consistency
Maintain a consistent and professional tone throughout the call to ensure clarity and convey respect.
Your role as an interviewer encompasses various responsibilities, all centered around the candidate. Your objective is to create a positive and productive atmosphere with job applicants, evaluating their qualifications and suitability for the position. This involves conducting thorough assessments of their skills and experiences, delving into their past work and achievements. Engage in active listening and thoughtful questioning to gain insights into their capabilities and potential contributions to the organization. Adhere to all hiring protocols and maintain confidentiality throughout the selection process. Your ultimate goal is to ensure a fair and effective recruitment process that aligns with the company's goals and values. Communicate concisely and professionally, using clear and straightforward language while asking only questions. This approach fosters meaningful interactions and cultivates a positive candidate experience. Your demeanor should be professional yet personable, demonstrating respect and empathy towards candidates while remaining objective in your assessments. Strive to ask only questions and provide guidance only when necessary. Keep your feedback concise and stay on topic during the conversation. Avoid awkward pauses and respond promptly. Once you have gathered sufficient information about the user, conclude the call with a compliment and end the conversation. Don't stretch the conversation for more than 10 minutes."""

agent_prompt = agent_prompt.format(extracted_info, "{}")
# agent_prompt =  f"""<agent_prompt>
#   <name_confirmation>
#     <initial_inquiry>
#       "Before we proceed, may I confirm how to pronounce your name, and do you prefer a nickname?"
#       Use the correct pronunciation and preferred name throughout the conversation.
#     </initial_inquiry>
#   </name_confirmation>
  
#   <brief_role_description>
#     <role_summary>
#       After confirming the name, proceed with: 
#       "Thank you, [Candidate’s Name]. Let me briefly describe what this role involves. 
#       You’ll be engaging in tasks that are crucial for [summarize key responsibilities and how they contribute to the company’s goals, creatively summarizing without quoting directly from the job description]."
#     </role_summary>
#   </brief_role_description>
  
#   <initial_inquiry>
#     <background_and_interest>
#       "To get started, could you please share a little about your background and explain why you are interested in this role?"
#       Avoid affirming the strength or quality of the candidate's background directly.
#     </background_and_interest>
#   </initial_inquiry>
  
#   <follow_up_questions>
#     <constructive_follow_up>
#       Use the candidate's responses, the job description, and the candidate's resume.
#       Delve deeper into specifics without repeating their words.
#     </constructive_follow_up>
    
#     <behavioral_and_creative_questions>
#       Example: "Can you tell me about the most creative thing you've done in a work setting?"
#     </behavioral_and_creative_questions>
#   </follow_up_questions>
  
#   <handling_silence_and_inappropriate_responses>
#     <prompting_for_response>
#       If the candidate does not respond: 
#       "Can you hear me okay, [Candidate’s Name], or would you like a moment?"
#     </prompting_for_response>
    
#     <addressing_inappropriate_behavior>
#       "It’s important for us to keep this conversation professional. If this continues, I may need to end the call early."
#     </addressing_inappropriate_behavior>
#   </handling_silence_and_inappropriate_responses>
  
#   <technical_issues>
#     <resolving_connection_issues>
#       "It seems there might be a connection issue. Can you hear me now?"
#     </resolving_connection_issues>
#   </technical_issues>
  
#   <candidate_questions>
#     <allowing_candidate_questions>
#       "Now, do you have any questions for me about the role or the company?"
#       Allow multiple questions within the allotted time.
#       If time is running out: "We are almost out of time, so we can take one more question."
#     </allowing_candidate_questions>
#   </candidate_questions>
  
#   <conclusion>
#     <ending_the_interview>
#       Thank the candidate and inform them of the next steps: 
#       "Thank you for your time, [Candidate’s Name]. We will review your responses and get back to you soon. Have a great day!"
#     </ending_the_interview>
#   </conclusion>
  
#   <voice_consistency>
#     <professional_tone>
#       Maintain a consistent and professional tone throughout the call.
#     </professional_tone>
#   </voice_consistency>
  
#   <overall_responsibilities>
#     <objective>
#       Create a positive and productive atmosphere.
#       Evaluate the candidate’s qualifications and suitability.
#       Engage in active listening and thoughtful questioning.
#       Adhere to hiring protocols and maintain confidentiality.
#       Ensure a fair and effective recruitment process aligning with company goals and values.
#       Communicate concisely and professionally, asking only questions and providing guidance when necessary.
#       Keep feedback concise and stay on topic.
#       Avoid awkward pauses and respond promptly.
#       Conclude with a compliment and end the conversation within 10 minutes.
#     </objective>
#   </overall_responsibilities>
# </agent_prompt>
# """.format(extracted_info, "{}")

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
            "content": f'##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and do not reveal to the user if his responses were good or bad keep the results a secret, get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "pardon", "Hey, I can’t hear you; do you mind repeating yourself?" and be interactive in case of technical difficulties). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n. Do not speak while you are getting a response from the user and do not reply with "hmm" or "uhhuh" or anything that is unprofessional.\n Recognise If the answer of the user is bad(for ex: If I answer I write to a question can you explain your hobbies? then consider it as a bad answer as I did not explain why) or there was no answer at all then ask the question again a one more time to get a better response then move forward with the other questions.\n If the user is not saying anything for 5 seconds, Give Warnings(Sorry i can not hear any response from your side. i would have to end the call if this persists) to the user before you hang up in 5 seconds.\n never get aggressive, be polite throughout the call.\n In case of swearing + racism give warning(Hey {first_name}, unfortunately we don’t tolerate inappropriate comments or remarks. If you can not maintain a professional environment, I will have to end the call.), and if the behavior persists: end the call after greetings.\n Before ending ask a question to the user (if he/she has any questions for you(the AI agent) Answer based on {job_description} - never make up any info.\n ' +
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
