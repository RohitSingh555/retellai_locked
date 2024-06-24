import os
import json
import chromadb
import ollama
import pdfplumber
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

def initialize_chromadb():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        print("ChromaDB connection initialized successfully.")
        return chroma
    except Exception as e:
        print(f"Failed to initialize ChromaDB connection: {e}")
        return None

def parse_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_resume_data(resume_text):
    # Simplified example for extracting resume data. Adjust with actual parsing logic.
    location = "Location: Example"  # Extract location from resume_text
    experience = "Experience: Example"  # Extract experience from resume_text
    skills = "Skills: Example"  # Extract skills from resume_text
    certificates = "Certificates: Example"  # Extract certificates from resume_text
    
    resume_data = {
        "location": location,
        "experience": experience,
        "skills": skills,
        "certificates": certificates
    }
    
    return resume_data

def document_exists(collection, doc_id):
    try:
        doc = collection.get(doc_id)
        return doc is not None
    except chromadb.exceptions.DocumentNotFound:
        return False
    except Exception as e:
        print(f"Error checking existence of document '{doc_id}': {e}")
        return False

def store_resume_and_transcript_data_in_chroma(resume_data, transcript_text, doc_id, collection_name, embedmodel='nomic-embed-text'):
    chroma = initialize_chromadb()
    if not chroma:
        return

    try:
        collection = chroma.get_or_create_collection(collection_name)
        print(f"Collection '{collection_name}' accessed or created successfully.")
    except Exception as e:
        print(f"Failed to access or create collection '{collection_name}': {e}")
        return

    try:
        if document_exists(collection, doc_id):
            print(f"Document '{doc_id}' already exists. Skipping.")
            return
        
        response = ollama.embeddings(model=embedmodel, prompt=f"Resume data: {json.dumps(resume_data)}")
        resume_embeddings = response["embedding"]
        
        transcript_embeddings = ollama.embeddings(model=embedmodel, prompt=f"Transcript data: {transcript_text}")["embedding"]
        
        collection.add(
            ids=[f"{doc_id}_resume", f"{doc_id}_transcript"],
            documents=[json.dumps(resume_data), transcript_text],
            embeddings=[resume_embeddings, transcript_embeddings],
            metadatas=[{"type": "resume"}, {"type": "transcript"}]
        )
        
        print(f"Data for document ID '{doc_id}' stored successfully with embeddings and metadata.")
    except Exception as e:
        print(f"Error storing data for document ID '{doc_id}': {e}")

def main():
    resumes_dir = "../transcripts/"
    transcripts_dir = "../transcripts/"
    collection_name = "resume_transcript_data"

    resumes = [f for f in os.listdir(resumes_dir) if f.endswith('.pdf')]
    transcripts = [f for f in os.listdir(transcripts_dir) if f.endswith('.pdf')]

    for resume_file in resumes:
        resume_number = ''.join(filter(str.isdigit, resume_file))
        related_transcript = next((t for t in transcripts if resume_number in t), None)
        
        if related_transcript:
            resume_text = parse_pdf(os.path.join(resumes_dir, resume_file))
            transcript_text = parse_pdf(os.path.join(transcripts_dir, related_transcript))
            
            resume_data = extract_resume_data(resume_text)
            doc_id = f"resume_{resume_number}"
            
            store_resume_and_transcript_data_in_chroma(resume_data, transcript_text, doc_id, collection_name)

if __name__ == "__main__":
    main()
