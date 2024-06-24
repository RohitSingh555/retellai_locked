import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import chromadb
import ollama
from bs4 import BeautifulSoup

load_dotenv()

def initialize_chromadb():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        print("ChromaDB connection initialized successfully.")
        return chroma
    except Exception as e:
        print(f"Failed to initialize ChromaDB connection: {e}")
        return None

def fetch_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Data fetched successfully from {url}")
        print(data)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data from {url}: {e}")
        return None

def document_exists(collection, doc_id):
    try:
        doc = collection.get(doc_id)
        return doc is not None
    except chromadb.exceptions.DocumentNotFound:
        return False
    except Exception as e:
        print(f"Error checking existence of document '{doc_id}': {e}")
        return False

def store_json_data_in_chroma(json_data, collection_name, embedmodel='nomic-embed-text'):
    chroma = initialize_chromadb()
    if not chroma:
        return

    try:
        collection = chroma.get_or_create_collection(collection_name)
        print(f"Collection '{collection_name}' accessed or created successfully.")
    except Exception as e:
        print(f"Failed to access or create collection '{collection_name}': {e}")
        return

    for item in json_data['products']:
        try:
            if 'body_html' in item:
                soup = BeautifulSoup(item['body_html'], 'html.parser')
                text_content = ', '.join(soup.stripped_strings)
                product_data = json.dumps({"body_html": text_content})

                for variant in item.get("variants", []):
                    variant_data = {
                        "title": item.get("title", ""),
                        "vendor": item.get("vendor", ""),
                        "product_type": item.get("product_type", ""),
                        "variant_title": variant.get("title", ""),
                        "price": variant.get("price", "")
                    }

                    doc_id = f'product_{item["id"]}_variant_{variant["id"]}'
                    if document_exists(collection, doc_id):
                        print(f"Document '{doc_id}' already exists. Skipping.")
                        continue

                    response = ollama.embeddings(model=embedmodel, prompt=f"Product data: {item['title']}, Variant: {variant['title']}")
                    embeddings = response["embedding"]

                    print(f"Generated document ID: {doc_id}")

                    collection.add(
                        ids=[doc_id],
                        documents=[product_data],
                        embeddings=[embeddings],
                        metadatas=[variant_data]
                    )
                    print(f"Data for product '{item['title']}' variant '{variant['title']}' stored successfully with embeddings and metadata.")
            else:
                print(f"Skipping product '{item['title']}' because 'body_html' is missing.")
        except KeyError as e:
            print(f"Missing data or invalid format in product '{item['title']}': {e}")
        except Exception as e:
            print(f"Error processing product '{item['title']}': {e}")

# Main function
def main():
    url = "https://jeffreestarcosmetics.com/products.json"
    collection_name = "jeffreestar1"

    json_data = fetch_data_from_url(url)
    if json_data:
        store_json_data_in_chroma(json_data, collection_name)

if __name__ == "__main__":
    main()
