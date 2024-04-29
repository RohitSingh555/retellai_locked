import os
import json
from openai import OpenAI
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_information_ai(text):
    api_key = "sk-proj-eZkmifxveCKsTRQT4KSfT3BlbkFJxhbZLPLS55Ai3cHSyEVC"
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
    }
    """

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt.replace("[Resume Text]", text),
        max_tokens=500,
        n=1,
        stop=["##"]
    )

    extracted_info = response.choices[0].text.strip()

    return extracted_info


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pdf_file = os.path.join(script_dir, "resume.pdf")

    resume_text = extract_text_from_pdf(pdf_file)

    extracted_info = extract_information_ai(resume_text)
    # print(extracted_info)

    # Convert extracted information to JSON format
    # extracted_info_json = json.loads(extracted_info)


    print("Extracted Information:",extracted_info)
