import os
import re
import openpyxl
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_information(text):
    # Define regular expressions for different categories
    personal_info_pattern = re.compile(r"Name: (.+)\nEmail: (.+)\nAddress: (.+)\nPhone: (.+)\nDate of birth: (.+)\nNationality: (.+)\nLink: (.+)", re.DOTALL)
    experience_pattern = re.compile(r"Experience\n(.+)\n([A-Za-z]+) - ([A-Za-z]+)\s+(\d{4}) - ([A-Za-z]+)\s+(\d{4})\n(.+)\n", re.DOTALL)
    education_pattern = re.compile(r"Education\n(.+)\n([A-Za-z]+) - ([A-Za-z]+)\s+(\d{4}) - ([A-Za-z]+)\s+(\d{4})\n(.+)\n", re.DOTALL)
    skills_pattern = re.compile(r"Languages Skills\n(.+)\n", re.DOTALL)
    projects_pattern = re.compile(r"Projects\n(.+)\n", re.DOTALL)
    introduction_pattern = re.compile(r"Introduction\n(.+)\n", re.DOTALL)
    certifications_pattern = re.compile(r"Certications & Courses\n(.+)\n", re.DOTALL)

    # Extract information using regular expressions
    personal_info = personal_info_pattern.search(text)
    experience = experience_pattern.search(text)
    education = education_pattern.search(text)
    skills = skills_pattern.search(text)
    projects = projects_pattern.search(text)
    introduction = introduction_pattern.search(text)
    certifications = certifications_pattern.search(text)

    extracted_info = {}

    if personal_info:
        extracted_info["personal_information"] = {
            "Name": personal_info.group(1),
            "Email": personal_info.group(2),
            "Address": personal_info.group(3),
            "Phone": personal_info.group(4),
            "Date of Birth": personal_info.group(5),
            "Nationality": personal_info.group(6),
            "Link": personal_info.group(7)
        }

    if experience:
        extracted_info["experience"] = {
            "Position": experience.group(1),
            "Company": experience.group(2),
            "Start Date": experience.group(3),
            "End Date": experience.group(4)
        }

    if education:
        extracted_info["education"] = {
            "Degree": education.group(1),
            "College": education.group(2),
            "Start Date": education.group(3),
            "End Date": education.group(4)
        }

    if skills:
        extracted_info["skills"] = skills.group(1).split()

    if projects:
        extracted_info["projects"] = projects.group(1).split("•")

    if introduction:
        extracted_info["introduction"] = introduction.group(1)

    if certifications:
        extracted_info["certifications"] = certifications.group(1).split("•")

    return extracted_info

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pdf_file = os.path.join(script_dir, "resumee.pdf")

    resume_text = extract_text_from_pdf(pdf_file)

    print(resume_text)
    print("Extracted Information:", resume_text)

