# populate_repository_graph.py

import os
import csv
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import sys

# Add src to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils import get_embedding_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def setup_mock_data():
    """Creates sample CSV and TXT files for demonstration."""
    print("Setting up mock data files...")
    
    # Mock UMLS data (Layer 3)
    umls_data = [
        ['C0011849', 'Diabetes Mellitus', 'Disease or Syndrome', 'A metabolic disorder characterized by hyperglycemia and insulin resistance.'],
        ['C0025540', 'Metformin', 'Pharmacologic Substance', 'A biguanide used in the treatment of type 2 diabetes.'],
        ['C0013604', 'Echocardiography', 'Diagnostic Procedure', 'A non-invasive imaging technique that uses ultrasound to visualize the heart.']
    ]
    with open('sample_umls.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cui', 'name', 'type', 'definition'])
        writer.writerows(umls_data)
        
    # Mock Paper data (Layer 2)
    paper_text = """
    A 2023 study in the Journal of Internal Medicine explored the long-term effects of Metformin on cardiovascular outcomes in patients with Type 2 Diabetes. The study concluded that continuous Metformin use was associated with a significant reduction in major adverse cardiovascular events. Echocardiography was used to assess cardiac function at baseline and follow-up, revealing improved diastolic function in the treatment group.

    Another review highlighted the importance of lifestyle modifications, such as diet and exercise, as a first-line treatment for newly diagnosed Type 2 Diabetes. While pharmacological interventions are crucial, they are most effective when combined with patient education and behavioral support.
    """
    with open('sample_papers.txt', 'w', encoding='utf-8') as f:
        f.write(paper_text)
    print("Mock data files 'sample_umls.csv' and 'sample_papers.txt' created.")

def create_gds_indexes(driver, dimension):
    """Creates GDS vector indexes for fast similarity search."""
    print("Creating GDS vector indexes...")
    with driver.session() as session:
        for index_name, label in [('umls-embeddings', 'UMLS_Entity'), ('paper-embeddings', 'Paper_Entity')]:
            try:
                session.run(f"""
                    CALL gds.vector.create(
                        '{index_name}',
                        '{label}',
                        'embedding',
                        {{
                            dimension: {dimension},
                            similarityFunction: 'cosine'
                        }}
                    )
                """)
                print(f"GDS index '{index_name}' created successfully.")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"GDS index '{index_name}' already exists. Dropping and recreating.")
                    session.run(f"CALL gds.graph.drop('{index_name}', false)")
                    session.run(f"""
                        CALL gds.vector.create(
                            '{index_name}',
                            '{label}',
                            'embedding',
                            {{
                                dimension: {dimension},
                                similarityFunction: 'cosine'
                            }}
                        )
                    """)
                    print(f"GDS index '{index_name}' recreated successfully.")
                else:
                    print(f"Error creating GDS index '{index_name}': {e}", file=sys.stderr)
                    raise e
    print("GDS index creation process finished.")


def ingest_umls_data(driver, embeddings, file_path='sample_umls.csv'):
    """Reads UMLS data from a CSV, generates embeddings, and stores it in Neo4j."""
    print("\n--- Starting UMLS Data Ingestion (Layer 3) ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_to_embed = f"Term: {row['name']}. Type: {row['type']}. Definition: {row['definition']}"
            embedding_vector = embeddings.embed_query(text_to_embed)
            
            with driver.session() as session:
                session.run("""
                    MERGE (u:UMLS_Entity {cui: $cui})
                    SET
                        u.name = $name,
                        u.type = $type,
                        u.definition = $definition,
                        u.embedding = $embedding
                """, cui=row['cui'], name=row['name'], type=row['type'], definition=row['definition'], embedding=embedding_vector)
            print(f"  Processed UMLS term: {row['name']}")
    print("--- UMLS Data Ingestion Complete ---")


def _extract_paper_entities(text, llm):
    """Helper to extract entities from paper text."""
    prompt = ChatPromptTemplate.from_template(
        """From the provided medical text, extract key entities.
        For each entity, provide its name, a general type (e.g., Condition, Treatment, Diagnostic Procedure), and a brief context.
        Respond in a valid JSON list format: [{"name": "...", "type": "...", "context": "..."}]
        
        Text:
        ---
        {text}
        ---
        """
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"text": text}).content
        cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error extracting paper entities: {e}", file=sys.stderr)
        return []

def ingest_papers_data(driver, embeddings, llm, file_path='sample_papers.txt'):
    """Reads paper text, extracts entities, generates embeddings, and stores in Neo4j."""
    print("\n--- Starting Medical Papers Ingestion (Layer 2) ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    entities = _extract_paper_entities(text, llm)
    
    if not entities:
        print("No entities extracted from paper text. Aborting paper ingestion.")
        return

    for entity in entities:
        entity_name = entity.get("name")
        if not entity_name:
            continue
            
        text_to_embed = f"Name: {entity_name}. Type: {entity.get('type')}. Context: {entity.get('context')}"
        embedding_vector = embeddings.embed_query(text_to_embed)
        
        with driver.session() as session:
            session.run("""
                MERGE (p:Paper_Entity {name: $name})
                SET
                    p.type = $type,
                    p.context = $context,
                    p.source_article = $source,
                    p.embedding = $embedding
            """, name=entity_name, type=entity.get('type'), context=entity.get('context'), source=file_path, embedding=embedding_vector)
        print(f"  Processed paper entity: {entity_name}")
    print("--- Medical Papers Ingestion Complete ---")


def main():
    """Main orchestration script."""
    load_dotenv()
    
    setup_mock_data()

    print("\nInitializing components (LLM, Embeddings, Neo4j)...")
    try:
        driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
        driver.verify_connectivity()
        embeddings = get_embedding_model()
        llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0)
    except Exception as e:
        print(f"FATAL: Failed to initialize components. Error: {e}")
        return
    print("Components initialized successfully.")

    print("\n" + "="*50)
    print("WARNING: This script will delete all existing data in the Neo4j database.")
    print("Make sure you have a backup if needed.")
    print("="*50)
    if input("Type 'yes' to continue: ").lower() != 'yes':
        print("Aborted by user.")
        return

    print("Clearing existing Neo4j database...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")

    ingest_umls_data(driver, embeddings)
    ingest_papers_data(driver, embeddings, llm)

    sample_embedding = embeddings.embed_query("test")
    dimension = len(sample_embedding)
    print(f"\nDetected embedding dimension: {dimension}")
    create_gds_indexes(driver, dimension)

    print("\n\n" + "*"*50)
    print("Repository Graph Population Complete!")
    print("Your Neo4j database is now populated with sample repository data and GDS indexes are created.")
    print("You can now run the main application.")
    print("*"*50)


if __name__ == "__main__":
    main()