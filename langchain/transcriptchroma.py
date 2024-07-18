import os
import json
import boto3
from dotenv import load_dotenv
import chromadb
import ollama
import time
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3', region_name='us-east-1')
bucket_name = 'sample-candidates-pluto-dev'
prefix = 'user_data_1/'

def initialize_chromadb():
    retries = 3
    delay = 2
    for attempt in range(retries):
        try:
            chroma = chromadb.HttpClient(host="localhost", port=7000)
            logger.info("ChromaDB connection initialized successfully.")
            return chroma
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: Failed to initialize ChromaDB connection: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries exceeded. Unable to initialize ChromaDB.")
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
            logger.error(f"Failed to list objects in S3: {e}")
            break

    return file_keys

def extract_and_store_data(file_key, bucket_name, collection_name):
    try:
        local_filename = file_key.split('/')[-1]
        s3.download_file(bucket_name, file_key, local_filename)
        
        with open(local_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if 'summary' not in json_data:
            logger.warning(f"Missing 'summary' key in {file_key}. Skipping this file.")
            os.remove(local_filename)
            return
        
        modified_data = {
            "summary": json_data["summary"],
            "strengths": json_data.get("strengths", ""),
            "weaknesses": json_data.get("weaknesses", ""),
            "cultural_fit": json_data.get("cultural_fit", ""),
            "decision": json_data.get("decision", "")
        }

        store_json_data_in_chroma(modified_data, collection_name)
        
        logger.info(f"Stored data from {file_key} in ChromaDB")
        
        os.remove(local_filename)
        
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {e}")

def convert_metadata(metadata):
    converted_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            converted_metadata[key] = ', '.join(value)
        else:
            converted_metadata[key] = str(value)
    return converted_metadata

def store_json_data_in_chroma(json_data, collection_name, embedmodel='nomic-embed-text'):
    chroma = initialize_chromadb()
    if not chroma:
        logger.error("ChromaDB not initialized. Unable to store data.")
        return
    
    try:
        collection = chroma.get_or_create_collection(collection_name)
        logger.info(f"Collection '{collection_name}' accessed or created successfully.")
    except Exception as e:
        logger.error(f"Failed to access or create collection '{collection_name}': {e}")
        return

    try:
        text_content = json_data['summary']
        metadata = {
            "strengths": json_data.get("strengths", ""),
            "weaknesses": json_data.get("weaknesses", ""),
            "cultural_fit": json_data.get("cultural_fit", ""),
            "decision": json_data.get("decision", "")
        }

        metadata = convert_metadata(metadata)

        response = ollama.embeddings(model=embedmodel, prompt=f"Product data: {text_content}")
        embeddings = response["embedding"]

        collection.add(
            ids=['example_id'],
            documents=[text_content],
            embeddings=[embeddings],
            metadatas=[metadata]
        )
        
        logger.info(f"Data stored successfully in ChromaDB with embeddings and metadata.")

    except Exception as e:
        logger.error(f"Error storing data in ChromaDB: {e}")

def main():
    collection_name = "sample_candidates_pluto_data"

    file_keys = fetch_data_from_s3(bucket_name, prefix)
    if not file_keys:
        logger.warning(f"No files found in S3 bucket {bucket_name}/{prefix}")
        return

    for file_key in file_keys:
        extract_and_store_data(file_key, bucket_name, collection_name)

if __name__ == "__main__":
    main()
