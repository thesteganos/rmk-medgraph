# tagging_pipeline.py

import os
import sys
import json
import argparse
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import project modules from the src package
from src.utils import get_embedding_model

MEDICAL_TAG_CATEGORIES = [
    "Primary Condition/Topic",
    "Symptoms & Signs",
    "Etiology & Risk Factors",
    "Pathophysiology",
    "Diagnostic Methods",
    "Treatment Modalities",
    "Preventive Measures",
    "Prognosis & Complications",
    "Epidemiology",
    "Relevant Anatomy",
    "Key Lab Results/Biomarkers"
]

def get_llm():
    """Initializes and returns the LLM."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    missing_vars = []
    if not google_api_key:
        missing_vars.append("GOOGLE_API_KEY")
    if not model_name:
        missing_vars.append("LLM_MODEL")

    if missing_vars:
        print(f"Error: The following environment variables are missing or not set in your .env file: {', '.join(missing_vars)}. Please ensure they are correctly configured.")
        sys.exit(1)

    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)

def get_neo4j_driver():
    """Initializes and returns the Neo4j driver."""
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    missing_vars = []
    if not uri:
        missing_vars.append("NEO4J_URI")
    if not user:
        missing_vars.append("NEO4J_USERNAME")
    if not password:
        missing_vars.append("NEO4J_PASSWORD")

    if missing_vars:
        print(f"Error: The following environment variables are missing or not set in your .env file: {', '.join(missing_vars)}. Please ensure they are correctly configured.")
        sys.exit(1)

    return GraphDatabase.driver(uri, auth=(user, password))

def tag_graphs_with_llm(driver: GraphDatabase.driver, llm: ChatGoogleGenerativeAI, embeddings):
    """Finds untagged MetaMedGraphs and generates structured tag summaries with embeddings."""
    print("--- Starting Graph Tagging Pipeline ---")
    tag_prompt_text = (
        "You are a medical expert system. Your task is to generate a concise yet comprehensive, structured summary for the provided 'Medical Content'. "
        "The 'Medical Content' is a collection of 'Entity: Context' pairs extracted from a specific document chunk.\n\n"  # Note: \n are for multiline in a single string
        "Your goal is to distill this content into a structured summary. For each category below, synthesize the information from the 'Medical Content' "
        "to provide a focused description or a list of the most salient key points. Prioritize information that is central to understanding the chunk's relevance to that category. "
        "If a category is not substantively addressed in the provided content, explicitly write 'N/A' for that category.\n\n"
        "Categories:\n"
        + "\n".join([f"- **{cat}:** [Focused description/salient key points or N/A]" for cat in MEDICAL_TAG_CATEGORIES]) +
        "\n\nStrictly adhere to this format. The output must be only the structured summary. Do not add any other explanatory text, remarks, or introductions.\n\n"
        "Medical Content (List of 'Entity: Context' pairs from a document chunk):\n"
        "---\n{content}\n---\n\n"
        "Structured Summary:"
    )
    tag_prompt_template = ChatPromptTemplate.from_template(tag_prompt_text)
    chain = tag_prompt_template | llm

    with driver.session() as session:
        result = session.run("""
            MATCH (g:MetaMedGraph)
            WHERE NOT (g)-[:HAS_TAG_SUMMARY]->()
            MATCH (g)<-[:PART_OF]-(e:RAG_Entity)
            RETURN g.chunk_id AS chunk_id, collect(e.name + ': ' + e.context) AS content
        """)

        for record in result:
            chunk_id = record["chunk_id"]
            content = "\n".join(record["content"])

            print(f"Tagging graph for chunk: {chunk_id}")
            try:
                tag_summary_text = chain.invoke({"content": content}).content
            except Exception as e:
                print(f"Error generating tag summary for chunk {chunk_id}: {e}")
                print("Skipping this chunk.")
                continue # Skip to the next record in the loop

            # Note: embedding_vector generation is dependent on successful tag_summary_text
            embedding_vector = embeddings.embed_query(tag_summary_text)

            try:
                session.run("""
                    MATCH (g:MetaMedGraph {chunk_id: $chunk_id})
                    MERGE (t:TagSummary {text: $summary_text})
                    SET t.embedding = $embedding
                    MERGE (g)-[:HAS_TAG_SUMMARY]->(t)
                """, chunk_id=chunk_id, summary_text=tag_summary_text, embedding=embedding_vector)
            except Exception as e:
                print(f"Error saving tag summary for chunk {chunk_id} to Neo4j: {e}")
                print("Skipping this chunk.")
                # If the LLM call was successful, tag_summary_text and embedding_vector will exist.
                continue

    print("--- Graph Tagging Pipeline Complete ---")

def _get_clusters_for_level(driver, level):
    """Fetches nodes and their embeddings from a specific level."""
    with driver.session() as session:
        if level == 0:
            query = """
                MATCH (t:TagSummary)
                WHERE NOT (t)<-[:SUMMARIZES]-()
                RETURN id(t) AS node_id, t.text AS text, t.embedding as embedding
            """
        else:
            query = """
                MATCH (t:AbstractTag {level: $level})
                WHERE NOT (t)<-[:SUMMARIZES]-()
                RETURN id(t) AS node_id, t.text AS text, t.embedding as embedding
            """
        # NOTE: The caller already passes in the desired level to
        # `_get_clusters_for_level`. Using `level-1` here unintentionally
        # shifts the query to the wrong hierarchy level, leading to empty
        # results or incorrect clustering behavior. We should query Neo4j
        # using the level provided by the caller.
        results = session.run(query, level=level)
        return [{"node_id": r["node_id"], "text": r["text"], "embedding": r["embedding"]} for r in results]

def _generate_new_summary(llm, child_texts):
    """Uses an LLM to create a new, more abstract summary from child summaries."""
    if len(child_texts) == 1: return child_texts[0]

    formatted_children = "\n".join(f"- {text}" for text in child_texts)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert medical ontologist. Your task is to synthesize a concise and descriptive parent topic summary from two distinct child topic summaries. "
        "This parent summary must be more abstract and general than the children. Crucially, it must accurately and specifically capture the primary shared concept or theme that links the two child summaries. "
        "Avoid vague generalizations. The goal is a parent topic that is both abstract and clearly indicative of the underlying commonality. "
        "Do not simply concatenate or list the child topics.\n\n"  # Note: \n for multiline
        "CHILD TOPIC SUMMARIES (each representing a distinct cluster of medical information):\n"
        "---\n{topics}\n---\n\n"
        "CONCISE, DESCRIPTIVE, AND ABSTRACT PARENT TOPIC SUMMARY:"
    )
    chain = prompt | llm
    try:
        return chain.invoke({"topics": formatted_children}).content
    except Exception as e:
        print(f"Error generating summary for topics '{formatted_children}': {e}")
        return None # Or consider re-raising if the caller must handle it more specifically

def _store_new_cluster(session, level, new_summary_text, new_embedding, child_node_ids):
    """Creates the new AbstractTag node and links it to its children in Neo4j."""
    result = session.run("""
        MERGE (p:AbstractTag {text: $text, level: $level})
        SET p.embedding = $embedding
        WITH p
        MATCH (c) WHERE id(c) IN $child_ids
        MERGE (p)-[:SUMMARIZES]->(c)
        RETURN id(p) as new_node_id
    """, text=new_summary_text, level=level, embedding=new_embedding, child_ids=child_node_ids)
    return result.single()["new_node_id"]

def build_tag_hierarchy(driver, llm, embeddings, max_levels=12, merge_percentile_threshold=0.75):
    """Applies hierarchical clustering to generate abstract tag summaries."""
    print("\n" + "="*50)
    print("--- Starting Tag Hierarchy Building ---")

    for level in range(1, max_levels + 1):
        print(f"\n--- Processing Level {level} ---")

        clusters_to_process = _get_clusters_for_level(driver, level - 1)

        if len(clusters_to_process) < 2:
            print(f"Not enough clusters ({len(clusters_to_process)}) to process further. Halting.")
            break

        print(f"Found {len(clusters_to_process)} clusters to process from level {level - 1}.")

        vectors = [c["embedding"] for c in clusters_to_process if c["embedding"]]
        if len(vectors) != len(clusters_to_process):
             print("Warning: Some clusters are missing embeddings. They will be skipped.")
             clusters_to_process = [c for c in clusters_to_process if c["embedding"]]

        if len(vectors) < 2:
            print("Not enough clusters with embeddings to compare. Halting level.")
            continue

        similarity_matrix = cosine_similarity(vectors)
        similarities = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]
        if len(similarities) == 0: break
        dynamic_threshold = np.percentile(similarities, merge_percentile_threshold * 100)
        print(f"Dynamic similarity threshold for merging: {dynamic_threshold:.4f}")

        merged_ids = set()

        pair_indices = np.triu_indices(len(clusters_to_process), k=1)
        sorted_pair_indices = sorted(zip(*pair_indices), key=lambda p: similarity_matrix[p[0], p[1]], reverse=True)

        with driver.session() as session:
            for i, j in sorted_pair_indices:
                if similarity_matrix[i, j] < dynamic_threshold: break

                cluster1_id = clusters_to_process[i]["node_id"]
                cluster2_id = clusters_to_process[j]["node_id"]

                if cluster1_id not in merged_ids and cluster2_id not in merged_ids:
                    print(f"  Merging cluster {cluster1_id} and {cluster2_id}...")
                    merged_ids.add(cluster1_id)
                    merged_ids.add(cluster2_id)

                    child_texts = [clusters_to_process[i]["text"], clusters_to_process[j]["text"]]
                    new_summary = _generate_new_summary(llm, child_texts)

                    if new_summary is None:
                        print(f"Skipping merge of clusters {cluster1_id} and {cluster2_id} due to summary generation failure.")
                        # Note: cluster1_id and cluster2_id were already added to merged_ids.
                        # This means they won't be promoted if this path is taken.
                        # For more robust error handling, merged_ids update should be after successful storage.
                        continue

                    new_embedding = embeddings.embed_query(new_summary)

                    try:
                        _store_new_cluster(session, level, new_summary, new_embedding, [cluster1_id, cluster2_id])
                    except Exception as e:
                        print(f"Error storing new cluster for children {cluster1_id}, {cluster2_id} (summary: '{new_summary}'): {e}")
                        # merged_ids already contains these IDs.
                        continue

            for i, cluster in enumerate(clusters_to_process):
                if cluster["node_id"] not in merged_ids:
                    print(f"  Promoting unmerged cluster {cluster['node_id']} to level {level}.")
                    _store_new_cluster(session, level, cluster["text"], cluster["embedding"], [cluster["node_id"]])

    print("--- Tag Hierarchy Building Complete ---")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tagging pipeline for MedGraphRAG. By default, only tags new graphs. Use --build-hierarchy to run the expensive hierarchy building process.")
    parser.add_argument("--build-hierarchy", action="store_true", help="Build the tag hierarchy. This is a slow and expensive process.")
    args = parser.parse_args()

    driver = get_neo4j_driver()
    llm = get_llm()
    embeddings = get_embedding_model()

    tag_graphs_with_llm(driver, llm, embeddings)

    if args.build_hierarchy:
        build_tag_hierarchy(driver, llm, embeddings)
    else:
        print("\nTo build the tag hierarchy, run the script with the --build-hierarchy flag.")
        print("Example: python tagging_pipeline.py --build-hierarchy")

    print("\nScript finished.")
