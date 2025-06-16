# ingest.py

import os
import sys
import json
import re
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Add src to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import get_embedding_model

load_dotenv()
DATA_PATH = "data"
DB_PATH = "db"
PROCESSED_LOG_FILE = os.path.join(DB_PATH, "processed_files.log")

def get_llm():
    """Initializes and returns the LLM for ingestion tasks."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    if not google_api_key or not model_name:
        raise ValueError("GOOGLE_API_KEY and LLM_MODEL must be set.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)

def get_neo4j_driver():
    """Initializes and returns the Neo4j driver."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        print("Warning: Neo4j environment variables not fully set.", file=sys.stderr)
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}", file=sys.stderr)
        return None

def semantic_chunker(text: str, llm: ChatGoogleGenerativeAI, max_chunk_size=1000) -> list[str]:
    """Groups paragraphs by semantic topic using an LLM."""
    print("Starting semantic chunking...")
    prompt = ChatPromptTemplate.from_template(
        "You are a text segmentation system. Does the 'New Paragraph' continue the topic from the 'Current Chunk'? "
        "Answer with only 'yes' or 'no'.\n\n"
        "Current Chunk:\n---\n{current_chunk}\n---\n\n"
        "New Paragraph:\n---\n{new_paragraph}\n---"
    )
    chain = prompt | llm
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if not p.strip(): continue
        if not current_chunk:
            current_chunk = p
            continue

        if len(current_chunk.split()) + len(p.split()) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = p
            continue
            
        response = chain.invoke({"current_chunk": current_chunk, "new_paragraph": p})
        if "yes" in response.content.lower():
            current_chunk += "\n\n" + p
        else:
            chunks.append(current_chunk)
            current_chunk = p
            
    if current_chunk: chunks.append(current_chunk)
    print(f"Semantic chunking complete. Created {len(chunks)} chunks.")
    return chunks

def extract_entities_from_chunk(chunk_text: str, llm: ChatGoogleGenerativeAI) -> list:
    """Extracts entities from a text chunk."""
    prompt = ChatPromptTemplate.from_template(
        """From the text, extract biomedical entities. For each, provide name, type (e.g., Condition, Treatment), and context.
        Text: {chunk}
        Respond in JSON format: [{"name": "...", "type": "...", "context": "..."}]"""
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"chunk": chunk_text}).content
        cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error extracting entities: {e}", file=sys.stderr)
        return []

def link_to_repository(rag_entity_name: str, driver: GraphDatabase.driver, embeddings, text_to_embed: str):
    """Finds and links a RAG_Entity to repository nodes using GDS vector similarity."""
    entity_embedding = embeddings.embed_query(text_to_embed)
    
    with driver.session() as session:
        # Link to Paper Repository (Layer 2)
        session.run("""
            CALL gds.vector.knn('paper-embeddings', $embedding, {topK: 1, similarityMetric: 'COSINE'})
            YIELD node, similarity
            WHERE similarity > 0.85
            WITH node AS paper_node
            MATCH (r:RAG_Entity {name: $rag_name})
            MERGE (r)-[:IS_REFERENCED_BY]->(paper_node)
        """, embedding=entity_embedding, rag_name=rag_entity_name)

        # Link to UMLS Vocabulary (Layer 3)
        session.run("""
            CALL gds.vector.knn('umls-embeddings', $embedding, {topK: 1, similarityMetric: 'COSINE'})
            YIELD node, similarity
            WHERE similarity > 0.90
            WITH node AS umls_node
            MATCH (r:RAG_Entity {name: $rag_name})
            MERGE (r)-[:HAS_DEFINITION_IN]->(umls_node)
        """, embedding=entity_embedding, rag_name=rag_entity_name)

def ingest_text_as_new_document(text: str, doc_id: str, llm, driver, embeddings):
    """Processes a single piece of text and adds it fully to the MedGraphRAG structure."""
    print(f"--- Ingesting new text for doc_id: {doc_id} ---")
    chunks = semantic_chunker(text, llm)
    
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            session.run("MERGE (g:MetaMedGraph {chunk_id: $chunk_id})", chunk_id=chunk_id)

            entities = extract_entities_from_chunk(chunk, llm)
            for entity in entities:
                entity_name = entity.get("name")
                if not entity_name: continue
                
                text_to_embed = f"Name: {entity_name}. Type: {entity.get('type')}. Context: {entity.get('context')}"
                embedding_vector = embeddings.embed_query(text_to_embed)
                
                session.run("""
                    MATCH (g:MetaMedGraph {chunk_id: $chunk_id})
                    MERGE (e:RAG_Entity {name: $name})
                    SET e.type = $type, e.context = $context, e.source_document = $doc_id, e.embedding = $embedding
                    MERGE (e)-[:PART_OF]->(g)
                """, chunk_id=chunk_id, name=entity_name, type=entity.get("type"), context=entity.get("context"), doc_id=doc_id, embedding=embedding_vector)
                
                link_to_repository(entity_name, driver, embeddings, text_to_embed)
    
    print(f"--- Finished ingesting text for doc_id: {doc_id} ---")

def get_processed_files():
    if not os.path.exists(PROCESSED_LOG_FILE): return set()
    with open(PROCESSED_LOG_FILE, "r", encoding='utf-8') as f:
        return set(line.strip() for line in f)

def log_processed_file(filename):
    with open(PROCESSED_LOG_FILE, "a", encoding='utf-8') as f:
        f.write(filename + "\n")

def main():
    """Main ingestion script for user-provided documents."""
    llm = get_llm()
    driver = get_neo4j_driver()
    embeddings = get_embedding_model()
    
    if not driver:
        print("Cannot proceed without Neo4j connection.", file=sys.stderr)
        return

    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.listdir(DATA_PATH):
        print(f"The '{DATA_PATH}' directory is empty. Please add PDF files to ingest.")
        return

    processed_files = get_processed_files()
    all_files = set(os.listdir(DATA_PATH))
    new_files = sorted(list(all_files - processed_files))

    if not new_files:
        print("No new documents to process.")
        return

    print(f"Found {len(new_files)} new document(s) to process: {new_files}")

    for filename in new_files:
        if not filename.lower().endswith('.pdf'): continue
        file_path = os.path.join(DATA_PATH, filename)
        try:
            print(f"\nProcessing '{filename}'...")
            loader = PyPDFLoader(file_path)
            document_text = "\n\n".join(p.page_content for p in loader.load())
            ingest_text_as_new_document(document_text, filename, llm, driver, embeddings)
            log_processed_file(filename)
            print(f"Successfully processed and logged '{filename}'.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()