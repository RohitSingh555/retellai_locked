import chromadb
import ollama
import json
from datetime import datetime

def initialize_chromadb():
    try:
        chroma = chromadb.HttpClient(host="localhost", port=8000)
        return chroma
    except ImportError:
        print("ChromaDB module not found.")
        return None

def retrieve_data_from_chromadb(embeddings, collection_name, filters):
    chroma = initialize_chromadb()
    if chroma:
        collection = chroma.get_collection(collection_name)
        if embeddings:
            result = collection.query(query_embeddings=[embeddings], n_results=2)
            # result = collection.query(query_embeddings=[embeddings], n_results=2,where_document={"$contains": "bag"})
            if result and result.get("documents"):
                document = result
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
    
    while True:
        user_input = input("You: ").strip().lower()

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
            "tags": {"$eq": "Accessory"}
        }

        document = retrieve_data_from_chromadb(embeddings, collection_name, filters)

        if document:
            chat_history.add_message("system", f"Here is the information I found: {document}")
            model_query = f"{chat_history.get_prompt()} Answer the last question using the following text as a resource and make sure that you provide details like a product seller. Keep your responses on point and within 30 words.\n: {document}"
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
    collection_name = "jeffreestar1"
    chatbot(collection_name)

if __name__ == "__main__":
    main()
