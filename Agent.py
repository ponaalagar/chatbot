from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from PyPDF2 import PdfReader 
from typing_extensions import List, TypedDict
import os

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["USER_AGENT"] = "demo/0.1"
os.environ["GROQ_API_KEY"] = "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"

# Initialize LLM and vector store
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)

# Load files (PDFs and texts) and extract content
def load_files(folder: str):
    documents = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        
        if filename.endswith(".txt"):
            # Handle text files
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
                print('finish','-'*10)
        
        elif filename.endswith(".pdf"):
            # Handle PDF files
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()  # Extract text from each page
                documents.append(Document(page_content=text, metadata={"source": filename}))
                print(f'Processed PDF file: {filename}')
    return documents

# Update the loader to work with PDFs and texts
data_folder = "./data"  # Folder containing PDFs and text files
docs = load_files(data_folder)

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add the document chunks to the vector store
_ = vector_store.add_documents(documents=all_splits)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define functions for retrieve and generate
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    # Combine the prompt with the question and context, ensuring proper formatting
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Construct the message to be passed to the LLM
    message = f"""
    Welcome to Aram Eyecare! We are dedicated to providing you with comprehensive eye care solutions for individuals of all ages. Whether you're looking for expert consultations, advanced diagnostics, or personalized treatments, we are here to help. Our services include:

    - Diagnosis and treatment of common eye conditions like Myopia, Amblyopia, Presbyopia, and early-stage glaucoma detection.
    - Solutions for Computer Vision Syndrome (CVS) and other modern eye-related problems.
    - Specialized services like contact lens fitting, foreign body removal, pre- and post-operative counseling, and more.
    - Eyewear dispensing and vision correction tailored to your needs.

    Our mission is to empower optometrists and offer top-notch eye care using innovative technologies. Feel free to ask us any questions regarding your eye health or our services. We're here to guide you toward a brighter, clearer future with optimal vision!

    --- 
    Question: {state["question"]}

    Context: {docs_content}
    If I don't know the answer to your question or if your question is out of the topic, I recommend contacting the Aram Eyecare team directly. You can reach them at:
    - Phone: +91 9600481947
    - Email: aramprimaryeyecare@gmail.com
    """
    
    try:
        response = llm.invoke(message)  # Pass the formatted message to the LLM
        return {"answer": response.content}
    except Exception as e:
        print(f"Error generating response: {e}")  # Log any error for troubleshooting
        return {"error": "Failed to generate response"}

# Build the state graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        response = graph.invoke({"question": question})
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
