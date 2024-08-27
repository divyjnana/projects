import os
import shutil
import argparse
import numpy as np
import PyPDF2
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import pickle

class Encoder:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy().tolist()
        return embedding

CHROMA_PATH = "chroma"
DATA_PATH = r"C:\Users\Hp\Downloads\PROJECTS\data"  # Update to the path where PDF files are stored
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

class Chroma:
    def __init__(self, persist_directory, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.documents = []

    def similarity_search_with_relevance_scores(self, query_text, k=3):
        query_embedding = self.embedding_function.embed_query(query_text)
        similarity_scores = []

        for document in self.documents:
            document_embedding = self.embedding_function.embed_query(document.page_content)
            similarity_score = self.calculate_cosine_similarity(query_embedding, document_embedding)
            similarity_scores.append((document, similarity_score))

        if not similarity_scores:
            print("No similarity scores found. Ensure documents are loaded and embeddings are computed correctly.")
            return []

        # Normalize similarity scores
        max_score = max(similarity_scores, key=lambda x: x[1])[1]
        min_score = min(similarity_scores, key=lambda x: x[1])[1]
        normalized_scores = [(doc, (score - min_score) / (max_score - min_score)) for doc, score in similarity_scores]

        # Filter results based on threshold
        filtered_scores = [(doc, score) for doc, score in normalized_scores if score >= 0.7]

        # Sort by similarity score in descending order
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return filtered_scores[:k]

    def calculate_cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
        return cosine_similarity

    def persist(self):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        with open(os.path.join(self.persist_directory, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Persisted {len(self.documents)} documents to {self.persist_directory}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["generate", "query", "evaluate"], required=True,
                        help="Mode to run the script in.")
    parser.add_argument("query_text", type=str, nargs='?', help="The query text.")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_data_store()
    elif args.mode == "query":
        if args.query_text is None:
            print("Error: query_text is required for query mode.")
            return
        query_database(args.query_text)
    elif args.mode == "evaluate":
        evaluate_embeddings()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    documents = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file_name)
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                documents.append(Document(page_content=text, metadata={"source": file_name}))
    print(f"Loaded {len(documents)} PDF documents.")
    return documents

def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: List[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_function = HuggingFaceEmbeddings()
    embeddings = embedding_function.embed_documents([chunk.page_content for chunk in chunks])
    print(f"Calculated embeddings for {len(chunks)} chunks.")
    
    chroma = Chroma(CHROMA_PATH, embedding_function)
    chroma.documents = chunks
    chroma.embedding_function = embedding_function
    chroma.persist_directory = CHROMA_PATH

    chroma.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

class HuggingFaceEmbeddings:
    def __init__(self):
        self.encoder = Encoder(embedding_model_name)

    def embed_query(self, text):
        return self.encoder.encode(text)

    def embed_documents(self, texts):
        embeddings = [self.encoder.encode(text) for text in texts]
        print(f"Calculated embeddings for {len(embeddings)} documents.")
        return embeddings
from transformers import BartTokenizer, BartForConditionalGeneration

def generate_response(context_text, query_text):
    # Use a summarization model like BART or T5
    summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    # Create a prompt for summarization
    prompt = f"Summarize the following context and answer the question.\n\nContext:\n{context_text}\n\nQuestion:\n{query_text}"
    
    # Tokenize the input
    inputs = summarization_tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
    
    # Generate the summary
    summary_ids = summarization_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def query_database(query_text):
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    if not os.path.exists(os.path.join(CHROMA_PATH, 'documents.pkl')):
        print("Error: No documents found in the Chroma database. Please run the script in 'generate' mode first.")
        return

    with open(os.path.join(CHROMA_PATH, 'documents.pkl'), 'rb') as f:
        db.documents = pickle.load(f)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(f"Context: {context_text}\nQuestion: {query_text}")

    response_text = generate_response(context_text, query_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


def evaluate_embeddings():
    embedding_function = HuggingFaceEmbeddings()

    # Take user input for the two words
    word1 = input("Enter the first word: ")
    word2 = input("Enter the second word: ")

    # Compute embeddings for both words
    vec1 = np.array(embedding_function.embed_query(word1))
    vec2 = np.array(embedding_function.embed_query(word2))

    # Calculate the distance between the embeddings
    distance = np.linalg.norm(vec1 - vec2)
    print(f"Comparing ({word1}, {word2}): {distance}")

if __name__ == "__main__":
    main()
