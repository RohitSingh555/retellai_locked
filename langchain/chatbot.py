import chromadb
import ollama
import json
from datetime import datetime
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import json
from docx import Document
import openai
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()

class JobAnalysis(BaseModel):
    company_name: str = Field(description='return a string output that summarizes the candidate')
    background_summary: str = Field(description='return a string output that answers the question in detail')
    job_description_summary: str = Field(description='return a string output that answers the question in detail')
    top_skills: list = Field(description='return a list with the top 3 skills required for the job')
                      
# job_description = "Senior Data Scientist at Moderna. This role involves developing and overseeing Immuno-Assay development, essential for Moderna's research and vaccine production efforts."
# company_background = "Moderna is a biotechnology company pioneering messenger RNA (mRNA) therapeutics and vaccines. We aim to transform how medicines are created and delivered, focusing on preventing and fighting diseases."

def job_description_analysis_ai():
    llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
    )
    # openai.api_key = os.getenv("OPENAI_ID")


    prompt = '''
    <Role>
    You are an AI assistant trained to analyze job descriptions.
    <Role>

    <Instructions>
    Your job is to answer the following questions, using the job description and company background provided to you. You must only return a JSON response with the following keys: company_name, background_summary, job_description_summary, top_skills
    </Instructions>

    <Questions>
    What is the company name?
    Summarize the company background and the company’s values in 1 paragraph.
    Summarize the job description in 1 paragraph.
    What are the 3 most important skills required for the job? Don't write generic skills, mention only technical skills in one word. 
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

    Json Response:

    '''
    parser = PydanticOutputParser(pydantic_object=JobAnalysis)

    prompt = PromptTemplate(
        template = prompt,
        input_variables = ["job_description", "company_background", "content"],
        partial_variables = {"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    return chain
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are an AI assistant trained to evaluate job candidates based on their resume and interview transcript."},
#         {"role": "user", "content": prompt}
#     ]
# )

# response_content = response.choices[0].message
# content_part = response_content.content
# print(content_part)

# with open('parsed_transcript5.json', 'w') as f:
#     f.write(content_part)

# print("Response saved to parsed_transcript5.json")



def initialize_chromadb():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        return chroma
    except ImportError:
        print("ChromaDB module not found.")
        return None

def retrieve_data_from_chromadb( collection_name, job_description, company_background, filters, k):
    chain = job_description_analysis_ai()
    response = chain.invoke({"job_description": job_description, "company_background": company_background})
    response_parsed = json.loads(response.json())
    job_description_analysis=f"{response_parsed}"
    print(job_description_analysis)
    user_prompt= "Suggest the top candidates suitable for the below job role:"
    final_prompt = user_prompt+" "+job_description_analysis
    response = ollama.embeddings(model='nomic-embed-text', prompt=final_prompt)
    embeddings = response.get("embedding", [])
    chroma = initialize_chromadb()
    if chroma:
        collection = chroma.get_collection(collection_name)
        if embeddings:
            result = collection.query(query_embeddings=[embeddings], n_results=k, where=None)
            # result = collection.query(query_embeddings=[embeddings], n_results=2,where_document={"$contains": "bag"})
            if result and result.get("documents"):
                document = result
                # print(document)
                return document
    return None

class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_prompt(self):
        prompt = ""
        for message in self.history:
            prompt += f'{message["role"]}: {message["content"]}\n'
        return prompt

def chatbot(collection_name):
    chat_history = ChatHistory()
    chat_history.add_message("system", "You are an expert assistant. Provide concise and accurate responses based on the provided information.")
    
    job_description_analysis={
  "company_name": "Moderna",
  "background_summary": "Moderna, founded in 2010, is a leading biotechnology company specializing in mRNA technology for developing medicines. They are committed to revolutionizing medicine through innovation, with a focus on creating a diverse pipeline of mRNA-based therapies across various diseases. Moderna values inclusivity, scientific excellence, and global health impact.",
  "job_description_summary": "Moderna is seeking a Senior Scientist to join their Immuno-Assays group, responsible for leading immunoassay development and validation across multiple therapeutic areas. The role involves designing and executing complex experiments, ensuring high-quality scientific principles in immunoassays, and contributing to clinical trials.",
  "top_skills": [
    "Immunoassay development and validation",
    "Scientific strategy implementation",
    "Ex vivo experimental design"
  ]
}
    while True:
        user_input = input("\nYou: ").strip().lower()
        print("\n")
        if user_input in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        chat_history.add_message("human", user_input)

        response = ollama.embeddings(model='nomic-embed-text', prompt=user_input)
        embeddings = response.get("embedding", [])

        if not embeddings:
            print("Chatbot: Sorry, I couldn't process your request. Please try again.")
            chat_history.add_message("chatbot", "Sorry, I couldn't process your request. Please try again.")
            continue

        filters = {
            "tags": {"$eq": "Data"}
        }

        document = retrieve_data_from_chromadb(embeddings, collection_name, filters)

        if document:
            chat_history.add_message("system", f"Here is the information I found: {document}")
            model_query = f"{chat_history.get_prompt()} Answer the last question based on this job analysis and the following text as a resource and make sure that you provide details. Keep your responses on point and within 100 words.\n: {document}"
            stream = ollama.generate(model='llama3', prompt=model_query, stream=True)
            response_text = ""
            for chunk in stream:
                if chunk.get("response"):
                    response_text += chunk['response']
                    print(chunk['response'], end='', flush=True)
            chat_history.add_message("chatbot", response_text)
        else:
            print("\nChatbot: That's all I could find. Please be a little more descriptive for accurate results.")
            chat_history.add_message("chatbot", "That's all I could find. Please be a little more descriptive for accurate results.")

# Main function
def main():
    
    collection_name = "sample_candidates_pluto_data"
    # chatbot(collection_name)
    # filters = {
    #         "strengths": {"$contains": "Data"},
    #         "strengths": {"$eq": "Data"},
    #     }
    
    job_description ="""
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
    """
    company_background = """
    About Moderna

    Since our founding in 2010, we have aspired to build the leading mRNA technology platform, the infrastructure to reimagine how medicines are created and delivered, and a world-class team. We believe in giving our people a platform to change medicine and an opportunity to change the world. 

    By living our mission, values, and mindsets every day, our people are the driving force behind our scientific progress and our culture. Together, we are creating a culture of belonging and building an organization that cares deeply for our patients, our employees, the environment, and our communities.

    We are proud to have been recognized as a Science Magazine Top Biopharma Employer, a Fast Company Best Workplace for Innovators, and a Great Place to Work in the U.S.

    As we build our company, we have always believed an in-person culture is critical to our success. Moderna champions the significant benefits of in-office collaboration by embracing a 70/30 work model. This 70% in-office structure helps to foster a culture rich in innovation, teamwork, and direct mentorship. Join us in shaping a world where every interaction is an opportunity to learn, contribute and make a meaningful impact.
    """

    
    response = retrieve_data_from_chromadb( collection_name, job_description, company_background, k=5, filters= None)
    print(json.dumps(response, indent=4))
    

if __name__ == "__main__":
    main()
