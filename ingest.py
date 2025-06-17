# ingest.py

import os
import sys
import json
import re
import asyncio
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Add src to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import get_embedding_model

load_dotenv()

MAX_CONCURRENT_GEMINI_CALLS = 5
try:
    val = os.getenv("MAX_CONCURRENT_GEMINI_CALLS")
    if val is not None:
        MAX_CONCURRENT_GEMINI_CALLS = int(val)
except ValueError:
    print(f"Warning: Invalid value for MAX_CONCURRENT_GEMINI_CALLS ('{os.getenv('MAX_CONCURRENT_GEMINI_CALLS')}'). Must be an integer. Using default: {MAX_CONCURRENT_GEMINI_CALLS}")

DEFAULT_PAPER_LINKING_THRESHOLD = 0.85
DEFAULT_UMLS_LINKING_THRESHOLD = 0.90

try:
    PAPER_LINKING_THRESHOLD = float(os.getenv("PAPER_LINKING_THRESHOLD", str(DEFAULT_PAPER_LINKING_THRESHOLD)))
except ValueError:
    print(f"Warning: Invalid value for PAPER_LINKING_THRESHOLD. Must be a float. Using default: {DEFAULT_PAPER_LINKING_THRESHOLD}")
    PAPER_LINKING_THRESHOLD = DEFAULT_PAPER_LINKING_THRESHOLD

try:
    UMLS_LINKING_THRESHOLD = float(os.getenv("UMLS_LINKING_THRESHOLD", str(DEFAULT_UMLS_LINKING_THRESHOLD)))
except ValueError:
    print(f"Warning: Invalid value for UMLS_LINKING_THRESHOLD. Must be a float. Using default: {DEFAULT_UMLS_LINKING_THRESHOLD}")
    UMLS_LINKING_THRESHOLD = DEFAULT_UMLS_LINKING_THRESHOLD

print(f"INFO: Paper linking threshold: {PAPER_LINKING_THRESHOLD}")
print(f"INFO: UMLS linking threshold: {UMLS_LINKING_THRESHOLD}")

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

async def semantic_chunker(text: str, llm: ChatGoogleGenerativeAI, max_chunk_size=1000) -> list[str]:
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

        response = await chain.ainvoke({"current_chunk": current_chunk, "new_paragraph": p})
        if "yes" in response.content.lower():
            current_chunk += "\n\n" + p
        else:
            chunks.append(current_chunk)
            current_chunk = p

    if current_chunk: chunks.append(current_chunk)
    print(f"Semantic chunking complete. Created {len(chunks)} chunks.")
    return chunks

async def extract_entities_from_chunk(chunk_text: str, llm: ChatGoogleGenerativeAI) -> list:
    """Extracts entities from a text chunk."""
    prompt = ChatPromptTemplate.from_template(
        """You are an expert biomedical entity extractor. Your task is to identify and extract relevant biomedical entities from the provided text chunk. For each entity, you must provide its name, a precise type, and the surrounding context from the text that supports its extraction.

**Instructions:**
1.  **Entity Types:** Focus on specific and relevant biomedical entities. Examples of entity types include (but are not limited to):
    *   `Condition` (e.g., Type 2 Diabetes, Hypertension)
    *   `Treatment` (e.g., Metformin, Lifestyle Modification, Surgical Procedure)
    *   `Diagnostic Procedure` (e.g., Echocardiography, Blood Test)
    *   `Pharmacologic Substance` (e.g., Semaglutide, Insulin)
    *   `Medical Device` (e.g., Glucose Meter, Stent)
    *   `Symptom` (e.g., Hyperglycemia, Fatigue)
    *   `Pathogen` (e.g., COVID-19 Virus, Streptococcus bacteria)
    *   `Gene/Protein` (e.g., BRCA1, Insulin Receptor)
    *   `Anatomical Structure` (e.g., Pancreas, Cardiovascular System)
    *   `Health Metric` (e.g., Blood Pressure, A1C Level)
2.  **Context:** The 'context' field should be a short, direct quote or a very close paraphrase from the text that clearly shows why the entity is mentioned and its relevance in the chunk. This helps verify the extraction.
3.  **Specificity:** Avoid extracting overly generic terms (e.g., "patients", "study", "health") unless they are part of a more specific concept within the text.
4.  **JSON Output:** Respond ONLY with a valid JSON list of objects. Each object must contain the keys "name", "type", and "context". Do not include any other text, explanations, or markdown formatting.

**Example:**
If the text is: "Metformin is often prescribed for Type 2 Diabetes to help control blood sugar levels. Some patients also report mild gastrointestinal discomfort."

A good JSON output would be:
```json
[
  {
    "name": "Metformin",
    "type": "Pharmacologic Substance",
    "context": "Metformin is often prescribed for Type 2 Diabetes to help control blood sugar levels."
  },
  {
    "name": "Type 2 Diabetes",
    "type": "Condition",
    "context": "Metformin is often prescribed for Type 2 Diabetes to help control blood sugar levels."
  },
  {
    "name": "gastrointestinal discomfort",
    "type": "Symptom",
    "context": "Some patients also report mild gastrointestinal discomfort."
  }
]
```

Text:
---
{chunk}
---

Respond in JSON format:
"""
    )
    chain = prompt | llm
    try:
        response = await chain.ainvoke({"chunk": chunk_text})
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "").strip()
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
            WHERE similarity > $paper_sim_threshold
            WITH node AS paper_node
            MATCH (r:RAG_Entity {name: $rag_name})
            MERGE (r)-[:IS_REFERENCED_BY]->(paper_node)
        """, embedding=entity_embedding, rag_name=rag_entity_name, paper_sim_threshold=PAPER_LINKING_THRESHOLD)

        # Link to UMLS Vocabulary (Layer 3)
        session.run("""
            CALL gds.vector.knn('umls-embeddings', $embedding, {topK: 1, similarityMetric: 'COSINE'})
            YIELD node, similarity
            WHERE similarity > $umls_sim_threshold
            WITH node AS umls_node
            MATCH (r:RAG_Entity {name: $rag_name})
            MERGE (r)-[:HAS_DEFINITION_IN]->(umls_node)
        """, embedding=entity_embedding, rag_name=rag_entity_name, umls_sim_threshold=UMLS_LINKING_THRESHOLD)

async def ingest_text_as_new_document(text: str, doc_id: str, llm, driver, embeddings):
    """Processes a single piece of text and adds it fully to the MedGraphRAG structure."""
    print(f"--- Ingesting new text for doc_id: {doc_id} ---")
    chunks = await semantic_chunker(text, llm)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_CALLS)

    async def process_chunk(chunk_text: str, chunk_id_suffix: int):
        nonlocal doc_id, llm, driver, embeddings # Allow access to outer scope variables
        chunk_id = f"{doc_id}_chunk_{chunk_id_suffix}"

        async with semaphore:
            entities = await extract_entities_from_chunk(chunk_text, llm)

        # Neo4j operations are synchronous, so they run outside the semaphore
        # It's important to ensure that DB operations don't block the event loop for too long.
        # For this exercise, we assume they are quick enough.
        # If Neo4j client library offers async versions, those should be preferred.
        with driver.session() as session:
            session.run("MERGE (g:MetaMedGraph {chunk_id: $chunk_id})", chunk_id=chunk_id)
            for entity in entities:
                entity_name = entity.get("name")
                if not entity_name: continue

                text_to_embed = f"Name: {entity_name}. Type: {entity.get('type')}. Context: {entity.get('context')}"
                # Embedding generation can be I/O bound if it's a remote call,
                # but here it's synchronous CPU-bound from langchain.
                # If it were async, it should also be outside the semaphore if it's not a Gemini call,
                # or inside if it is a Gemini call and needs to be limited.
                # Assuming get_embedding_model provides a synchronous embedding method.
                embedding_vector = embeddings.embed_query(text_to_embed)

                session.run("""
                    MATCH (g:MetaMedGraph {chunk_id: $chunk_id})
                    MERGE (e:RAG_Entity {name: $name})
                    SET e.type = $type, e.context = $context, e.source_document = $doc_id, e.embedding = $embedding
                    MERGE (e)-[:PART_OF]->(g)
                """, chunk_id=chunk_id, name=entity_name, type=entity.get("type"), context=entity.get("context"), doc_id=doc_id, embedding=embedding_vector)

                # link_to_repository involves DB calls, keep it synchronous for now
                link_to_repository(entity_name, driver, embeddings, text_to_embed)
        print(f"Finished processing chunk {chunk_id}")

    tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    await asyncio.gather(*tasks)

    print(f"--- Finished ingesting text for doc_id: {doc_id} ---")

def get_processed_files():
    if not os.path.exists(PROCESSED_LOG_FILE): return set()
    with open(PROCESSED_LOG_FILE, "r", encoding='utf-8') as f:
        return set(line.strip() for line in f)

def log_processed_file(filename):
    with open(PROCESSED_LOG_FILE, "a", encoding='utf-8') as f:
        f.write(filename + "\n")

async def main():
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

    ingestion_tasks = []
    for filename in new_files:
        if not filename.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {filename}")
            continue
        file_path = os.path.join(DATA_PATH, filename)
        try:
            print(f"\nPreparing to process '{filename}'...")
            loader = PyPDFLoader(file_path)
            # Document loading itself can be I/O bound, consider running in executor if it becomes a bottleneck
            document_text = "\n\n".join(p.page_content for p in loader.load())
            # Create a task for each document ingestion
            ingestion_tasks.append(ingest_text_as_new_document(document_text, filename, llm, driver, embeddings))
        except Exception as e:
            print(f"Error preparing file {filename} for ingestion: {e}", file=sys.stderr)

    # Run all ingestion tasks concurrently
    results = await asyncio.gather(*ingestion_tasks, return_exceptions=True)

    for filename, result in zip(new_files, results):
        if isinstance(result, Exception):
            print(f"Error processing file {filename}: {result}", file=sys.stderr)
        else:
            # Assuming ingest_text_as_new_document doesn't return specific status,
            # success is implied if no exception was raised.
            log_processed_file(filename)
            print(f"Successfully processed and logged '{filename}'.")

if __name__ == "__main__":
    asyncio.run(main())
