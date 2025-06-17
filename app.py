# app.py

import asyncio
import sys
import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

def perform_startup_checks():
    load_dotenv()
    required_vars = ["GOOGLE_API_KEY", "LLM_MODEL", "NEO4J_URI"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(
            f"FATAL ERROR: Missing environment variables: {', '.join(missing_vars)}. "
            "Please configure your .env file."
        )
        return False
        
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
        with driver.session() as session:
            result = session.run("CALL gds.graph.exists('umls-embeddings') YIELD exists")
            if not result.single()['exists']:
                st.error(
                    "FATAL ERROR: Neo4j GDS vector index 'umls-embeddings' not found. "
                    "Please run the one-time setup script: `python populate_repository_graph.py`"
                )
                return False
    except Exception as e:
        st.error(f"FATAL ERROR: Could not connect to or verify Neo4j database. Error: {e}")
        return False
        
    return True

st.set_page_config(page_title="MedGraphRAG AI", page_icon="⚕️")

if not perform_startup_checks():
    st.stop()

from src.graph import WeightManagementGraph

st.title("⚕️ MedGraphRAG: Advanced Medical AI")

@st.cache_resource
def get_graph():
    print("INFO: Initializing MedGraphRAG agent...")

    default_context_limit = 7
    try:
        context_limit_str = os.getenv("INITIAL_CONTEXT_LIMIT")
        if context_limit_str:
            initial_context_limit = int(context_limit_str)
            if initial_context_limit <= 0:
                print(f"Warning: INITIAL_CONTEXT_LIMIT environment variable ('{context_limit_str}') must be a positive integer. Using default: {default_context_limit}.")
                initial_context_limit = default_context_limit
        else:
            initial_context_limit = default_context_limit
    except ValueError:
        # Ensure context_limit_str is defined for the warning message, even if os.getenv returned None
        context_limit_str_for_warning = context_limit_str if 'context_limit_str' in locals() else "None"
        print(f"Warning: Invalid value for INITIAL_CONTEXT_LIMIT environment variable ('{context_limit_str_for_warning}'). Must be an integer. Using default: {default_context_limit}.")
        initial_context_limit = default_context_limit

    print(f"INFO: MedGraphRAG agent will be initialized with num_initial_entities = {initial_context_limit}")
    graph_builder = WeightManagementGraph(num_initial_entities=initial_context_limit)
    return graph_builder.compile_graph()

def log_feedback(interaction: dict, feedback: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback,
        "interaction": interaction
    }
    with open("feedback_for_review.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + "\n")

app = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MedGraphRAG, an AI assistant designed for reliable and evidence-based medical information. How can I help you today?"}
    ]

with st.sidebar:
    st.header("Your Profile (Optional)")
    st.info("Providing this information can help generate more relevant answers.")
    user_profile = {
        "age": st.text_input("Age", ""),
        "sex": st.selectbox("Sex", ["", "Male", "Female", "Prefer not to say"]),
        "goal": st.text_input("Primary Goal (e.g., lose fat, manage condition)", "")
    }

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about medical conditions, treatments, or nutrition..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph_input = {"query": prompt, "user_profile": user_profile}
            final_answer = "Sorry, an unexpected error occurred."
            try:
                response = app.invoke(graph_input)
                if response and isinstance(response, dict):
                    final_answer = response.get("final_answer", "Sorry, an error occurred in the graph.")
                elif response:
                    final_answer = str(response)
            except Exception as e:
                st.error(f"An application error occurred: {e}")
                print(f"ERROR during app.invoke: {e}", file=sys.stderr)
            
            st.markdown(final_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    
    st.session_state.last_interaction = {
        "query": prompt,
        "answer": final_answer,
        "user_profile": user_profile
    }

if "last_interaction" in st.session_state:
    st.markdown("---")
    col1, col2, _ = st.columns([1, 2, 5])
    with col1:
        if st.button("👍 Good"):
            log_feedback(st.session_state.last_interaction, "good")
            st.success("Feedback saved! Thank you.")
    with col2:
        if st.button("👎 Bad"):
            log_feedback(st.session_state.last_interaction, "bad")
            st.error("Feedback logged for review. Thank you.")