import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

class ChatAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def respond(self, message, context=""):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. {context}"},
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class RAGAgent:
    def __init__(self, collection_name="docs", persist_dir="./chroma_db"):
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = self.chroma.get_or_create_collection(collection_name, embedding_function=self.ef)
        self.doc_count = self.collection.count()
    
    def add_documents(self, texts, ids=None):
        try:
            if not ids:
                ids = [f"doc_{self.doc_count + i}" for i in range(len(texts))]
            self.collection.upsert(documents=texts, ids=ids)
            self.doc_count = self.collection.count()
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def search(self, query, n=3):
        try:
            results = self.collection.query(query_texts=[query], n_results=n)
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Error searching: {e}")
            return []

class MultiAgentSystem:
    def __init__(self):
        self.chat_agent = ChatAgent()
        self.rag_agent = RAGAgent()
    
    def process(self, query, use_rag=True):
        context = ""
        if use_rag:
            docs = self.rag_agent.search(query)
            context = f"Relevant context: {' '.join(docs)}" if docs else ""
        
        return self.chat_agent.respond(query, context)
