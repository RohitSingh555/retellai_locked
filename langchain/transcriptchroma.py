import os
import json
import boto3
import uuid
from bs4 import BeautifulSoup
import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client('s3', region_name='us-east-1')
bucket_name = 'sample-candidates-pluto-dev'
prefix = 'user_data/'

def initialize_chromadb():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        print("ChromaDB connection initialized successfully.")
        return chroma
    except Exception as e:
        print(f"Failed to initialize ChromaDB connection: {e}")
        return None

def fetch_data_from_s3(bucket_name, prefix):
    file_keys = []
    continuation_token = None

    while True:
        try:
            if continuation_token:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
            else:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            
            if 'Contents' not in response:
                break
            
            file_keys.extend([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')])
            
            if response.get('IsTruncated'):
                continuation_token = response['NextContinuationToken']
            else:
                break
        except Exception as e:
            print(f"Failed to list objects in S3: {e}")
            break

    return file_keys

def extract_and_store_data(file_key, bucket_name, collection_name, chroma):
    try:
        local_filename = file_key.split('/')[-1]
        s3.download_file(bucket_name, file_key, local_filename)
        
        with open(local_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if 'summary' not in json_data:
            print(f"Missing 'summary' key in {file_key}. Skipping this file.")
            os.remove(local_filename)
            return

        required_keys = ["summary", "strengths", "weaknesses", "cultural_fit", "decision"]
        for key in required_keys:
            if key not in json_data:
                print(f"Missing '{key}' key in {file_key}. Skipping this file.")
                os.remove(local_filename)
                return

        modified_data = {
            "summary": json_data["summary"],
            "strengths": json_data["strengths"],
            "weaknesses": json_data["weaknesses"],
            "cultural_fit": json_data["cultural_fit"],
            "decision": json_data["decision"]
        }

        store_json_data_in_chroma(modified_data, collection_name, chroma)
        
        print(f"Stored data from {file_key} in ChromaDB")
        
        os.remove(local_filename)
        
    except json.JSONDecodeError as e:
        print(f"Error processing file {file_key}: {e}")
        os.remove(local_filename)
    except Exception as e:
        print(f"Error processing file {file_key}: {e}")

def convert_metadata(metadata):
    converted_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            converted_metadata[key] = ', '.join(value)
        else:
            converted_metadata[key] = str(value)
    return converted_metadata

def store_json_data_in_chroma(json_data, collection_name, chroma, embedmodel='nomic-embed-text'):
    if not chroma:
        return
    
    try:
        collection = chroma.get_or_create_collection(collection_name)
        print(f"Collection '{collection_name}' accessed or created successfully.")
    except Exception as e:
        print(f"Failed to access or create collection '{collection_name}': {e}")
        return

    try:
        text_content = json_data['summary']
        metadata = {
            "strengths": json_data["strengths"],
            "weaknesses": json_data["weaknesses"],
            "cultural_fit": json_data["cultural_fit"],
            "decision": json_data["decision"]
        }

        metadata = convert_metadata(metadata)

        response = ollama.embeddings(model=embedmodel, prompt=f"Product data: {text_content}")
        embeddings = response["embedding"]

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text_content],
            embeddings=[embeddings],
            metadatas=[metadata]
        )
        
        print(f"Data stored successfully in ChromaDB with embeddings and metadata.")

    except Exception as e:
        print(f"Error storing data in ChromaDB: {e}")

def main():
    chroma = initialize_chromadb()
    if not chroma:
        print("Failed to initialize ChromaDB.")
        return
    
    collection_name = "sample_candidates_pluto_data"

    file_keys = fetch_data_from_s3(bucket_name, prefix)
    if not file_keys:
        print(f"No files found in S3 bucket {bucket_name}/{prefix}")
        return

    for file_key in file_keys:
        extract_and_store_data(file_key, bucket_name, collection_name, chroma)

if __name__ == "__main__":
    main()
