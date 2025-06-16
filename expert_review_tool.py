# expert_review_tool.py

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import sys

# Add src to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ingest import ingest_text_as_new_document, get_neo4j_driver, get_llm
from utils import get_embedding_model

DB_PATH = "db"
PENDING_REVIEW_FILE = "pending_review.jsonl"
PROCESSED_DIR = "processed_logs"

def process_propositions(driver, llm, embeddings):
    """
    Processes new knowledge, and on approval, ingests it into the full
    MedGraphRAG structure.
    """
    print("\n--- Processing Automated Knowledge Propositions ---")
    if not os.path.exists(PENDING_REVIEW_FILE):
        print("No pending propositions file found.")
        return 0

    propositions = []
    with open(PENDING_REVIEW_FILE, "r", encoding='utf-8') as f:
        for line in f:
            try:
                propositions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            
    if not propositions:
        print("No new propositions to review.")
        return 0

    approved_count = 0
    for i, prop in enumerate(propositions):
        print(f"\n{'='*80}\nPROPOSITION {i+1}/{len(propositions)}\nSOURCE: {prop.get('source_url', 'N/A')}\n\nAI-Generated Question:\n{prop['question']}\n\nAI-Generated Answer:\n{prop['answer']}\n{'='*80}")
        action = input("Action: [a]pprove, [s]kip, [q]uit? ").lower()
        if action == 'q': break
        if action == 'a':
            text_to_add = f"Question: {prop['question']}\n\nVerified Answer: {prop['answer']}"
            doc_id = f"expert_approved_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            try:
                ingest_text_as_new_document(text_to_add, doc_id, llm, driver, embeddings)
                print(f"--> Approved and ingested into knowledge graph.")
                approved_count += 1
            except Exception as e:
                print(f"--> FAILED to ingest approved proposition: {e}", file=sys.stderr)
        else:
            print("--> Skipped.")
    
    if os.path.exists(PENDING_REVIEW_FILE):
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        archive_name = os.path.join(PROCESSED_DIR, f"propositions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        os.rename(PENDING_REVIEW_FILE, archive_name)
    
    return approved_count

def main():
    """Main function for the expert review tool."""
    load_dotenv()
    
    print("--- Expert Knowledge Verification & Ingestion Tool ---")
    driver = get_neo4j_driver()
    llm = get_llm()
    embeddings = get_embedding_model()
    
    if not all([driver, llm, embeddings]):
        print("FATAL: Could not initialize Neo4j, LLM, or embeddings. Exiting.", file=sys.stderr)
        return

    prop_added = process_propositions(driver, llm, embeddings)
    
    if prop_added > 0:
        print(f"\n--- WORKFLOW COMPLETE ---")
        print(f"Knowledge base updated with {prop_added} new document(s)!")
        print("IMPORTANT: Run the `tagging_pipeline.py` to process the new graphs into the hierarchy.")
    else:
        print("\n--- WORKFLOW COMPLETE ---")
        print("No new knowledge was added to the database.")

if __name__ == "__main__":
    main()