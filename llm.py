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

Company_name = "Moderna"
job_role = "Senior Data Scientist"
job_description="""
The Role:

Joining Moderna offers the unique opportunity to be part of a pioneering team that's revolutionizing medicine through mRNA technology with a diverse pipeline of development programs across various diseases. As an employee, you'll be part of a continually growing organization working alongside exceptional colleagues and strategic partners worldwide, contributing to global health initiatives. Moderna's commitment to advancing the technological frontier of mRNA medicines ensures a challenging and rewarding career experience with the potential to make a significant impact on patients' lives worldwide.

Moderna's strategic partnership with the UK Government is exemplified by our innovative presence at Harwell. Our mission is to establish a leading-edge research, development, and manufacturing facility as part of a long-term commitment to onshore mRNA vaccine production for respiratory diseases. This initiative will create a multitude of highly skilled jobs and foster collaboration with academic and NHS partners across the UK. We're looking for global experts eager to join us in this endeavor, contributing to a future where access to life-saving vaccines is a reality for all​​.

Moderna is seeking a talented, experienced, and motivated Senior Scientist to join the Immuno-Assays group to serve as a scientific expert and point of contact for the oversight of Immuno-Assay development, qualification/validation to support programs across multiple therapeutic areas. The Senior Scientist will lead a team in the design and execution of immunoassay development, qualification or validation, and the running of clinical samples to return high-quality data for clinical trials conducted by Moderna.

Here's What You’ll Do:

Your key responsibilities will be:

Providing scientific strategies to implement new immune assays (MSD, ligand binding, Multi/single plex assays).

Serving as a lead in the execution of multiple projects requiring immunoassay expertise.

Formulating novel solutions for the design of complex experiments, including ex vivo assays.

Demonstrating technical excellence and advancing high-quality scientific principles in immunoassays.

Conducting ex vivo experimental work for functional profiling of the immune system.

Performing quality review of experimental reports and ensuring staff compliance with safety and regulatory guidelines.

Your responsibilities will also include:

Researching scientific and technical literature to propose and implement innovative solutions applicable to the Laboratory.

Assessing staff strengths and development needs, assigning projects accordingly.

Identifying and resolving quality issues and performing quality review of study reports.

Attending technical conferences and exhibits as required.

The key Moderna Mindsets you’ll need to succeed in the role:

Pursue options in parallel: Your role will require effectively managing multiple tasks and projects concurrently, ensuring the best outcomes through a comprehensive approach.

Dynamic range: Demonstrating the capacity to drive both strategy and execution, adapting swiftly to new data and changing priorities.

Here’s What You’ll Bring to the Table:

Ph.D. (2+ years in immunology or related field) or MSc with more than 4-6 years’ experience in conducting assay validation in clinical Immunology under GLP/GCLP environment.

A background in infections in diseases or oncology and experience with handling laboratory pathogens as containment level 3 is an advantage.

Experience wilt single and multi-plex assays including MSD platform.

Strong level of understanding and expertise in design and executing immune-related studies and assays

The ability to work in a cross-functional work environment is critical; strong leadership skills and independence skills are expected.

Excellent written, presentation and interpersonal communication skills; ability to influence and collaborate in a team environment and with external stakeholders.

Candidate will be curious in exploring new technology, bold in proposing creative experimental designs and ideas. Will work collaboratively with multifunctional teams and will be relentless in pursuing successful outcomes

Possess strong computational skills, preferably experienced with Word, Excel, Power Point, GraphPad Prism

Knowledge of system software for data analysis and statistical analysis.

Experience of working in a regulated environment GLP, GCP, ISO standards is preferred.

Moderna offers personalized benefit programs and well-being resources as unique as our global workforce so employees can do their best work.

We recognize and appreciate your diverse needs and interests and do our best to support you at work and at home with:

Quality healthcare and insurance benefits
Lifestyle Spending Accounts to create your own pathway to well-being
Free premium access to fitness, nutrition, and mindfulness classes
Family planning and adoption benefits
Generous paid time off, including vacation, bank holidays, volunteer days, sabbatical, and a discretionary year-end shutdown
Educational resources
Savings and investments
Location-specific perks and extras!
The benefits offered may vary depending on the nature of your employment with Moderna and the country where you work.

About Moderna

Since our founding in 2010, we have aspired to build the leading mRNA technology platform, the infrastructure to reimagine how medicines are created and delivered, and a world-class team. We believe in giving our people a platform to change medicine and an opportunity to change the world. 

By living our mission, values, and mindsets every day, our people are the driving force behind our scientific progress and our culture. Together, we are creating a culture of belonging and building an organization that cares deeply for our patients, our employees, the environment, and our communities.

We are proud to have been recognized as a Science Magazine Top Biopharma Employer, a Fast Company Best Workplace for Innovators, and a Great Place to Work in the U.S.

As we build our company, we have always believed an in-person culture is critical to our success. Moderna champions the significant benefits of in-office collaboration by embracing a 70/30 work model. This 70% in-office structure helps to foster a culture rich in innovation, teamwork, and direct mentorship. Join us in shaping a world where every interaction is an opportunity to learn, contribute and make a meaningful impact.
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

