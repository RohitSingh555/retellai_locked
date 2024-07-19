import os
import openai
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from datetime import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def initialize_chromadb():
    try:
        chroma_db = Chroma(persist_directory="data", embedding_function=openai.embeddings, collection_name="lc_chroma_demo")
        print("ChromaDB connection initialized successfully.")
        return chroma_db
    except Exception as e:
        print(f"Failed to initialize ChromaDB connection: {e}")
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
    # Load a PDF document and split it into sections
    loader = PyPDFLoader("data/document.pdf")
    docs = loader.load_and_split()

    # Initialize the OpenAI chat model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Load the Chroma database from disk
    chroma_db = Chroma(persist_directory="data", embedding_function=embeddings, collection_name="lc_chroma_demo")

    # Get the collection from the Chroma database
    collection = chroma_db.get()

    # If the collection is empty, create a new one
    if len(collection['ids']) == 0:
        # Create a new Chroma database from the documents
        chroma_db = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory="data",
            collection_name="lc_chroma_demo"
        )

        # Save the Chroma database to disk
        chroma_db.persist()

    chat_history = ChatHistory()
    chat_history.add_message("system", "You are an expert assistant. Provide concise and accurate responses based on the provided information.")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        print("\n")
        if user_input in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        chat_history.add_message("human", user_input)

        query = user_input

        # Prepare the query
        print('Similarity search:')
        print(chroma_db.similarity_search(query))

        print('Similarity search with score:')
        print(chroma_db.similarity_search_with_score(query))

        # Add a custom metadata tag to the first document
        docs[0].metadata = {
            "tag": "demo",
        }

        # Update the document in the collection
        chroma_db.update_document(
            document=docs[0],
            document_id=collection['ids'][0]
        )

        # Find the document with the custom metadata tag
        collection = chroma_db.get(where={"tag": "demo"})

        # Execute the chain
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=chroma_db.as_retriever())
        response = chain(query)

        # Print the response
        response_text = response['result']
        print(f"Chatbot: {response_text}")
        chat_history.add_message("chatbot", response_text)

# Main function
def main():
    collection_name = "sample_candidates_pluto_data"
    chatbot(collection_name)

if __name__ == "__main__":
    main()
