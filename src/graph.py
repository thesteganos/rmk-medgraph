# src/graph.py

import os
import json
import sys
from typing import TypedDict, Literal
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from .utils import get_embedding_model
from .tools import pubmed_tool
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

class GraphState(TypedDict):
    query: str
    user_profile: dict
    query_type: Literal["foundational", "protocol", "hybrid", "unsafe"]
    documents: list
    web_results: list
    neo4j_results: list
    final_answer: str
    disclaimer_needed: bool

class WeightManagementGraph:
    def __init__(self, num_initial_entities: int = 7):
        load_dotenv()

        self.num_initial_entities = num_initial_entities
        print(f"INFO: Number of initial entities for context retrieval set to: {self.num_initial_entities}")

        # Existing INITIAL_CONTEXT_LIMIT from .env will be overridden by the parameter if provided,
        # but we can keep its loading logic if it serves other purposes or as a fallback if
        # the class were instantiated without args (though default is now provided).
        # For this task, the direct parameter `num_initial_entities` takes precedence.
        default_env_limit = 5 # Original default if .env var was used
        try:
            env_limit = int(os.getenv("INITIAL_CONTEXT_LIMIT", str(default_env_limit)))
            if env_limit <= 0:
                print(f"Warning: INITIAL_CONTEXT_LIMIT in .env must be positive. Using default: {default_env_limit} if not overridden by parameter.")
                # self.num_initial_entities could be set based on this if no param was passed,
                # but the new __init__ signature with a default makes this less direct.
            # If num_initial_entities wasn't the default 7, it means it was explicitly passed.
            # If it's 7 (default), we could consider using env_limit, but the task implies
            # the new parameter is the primary control. Let's prioritize the new parameter.
            if self.num_initial_entities == 7 and os.getenv("INITIAL_CONTEXT_LIMIT"): # if default is used and env var exists
                 print(f"INFO: INITIAL_CONTEXT_LIMIT from .env ('{env_limit}') is also present. The constructor parameter/default ('{self.num_initial_entities}') takes precedence.")

        except ValueError:
            print(f"Warning: Invalid value for INITIAL_CONTEXT_LIMIT in .env. Must be an integer.")

        google_api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("LLM_MODEL")
        if not google_api_key or not model_name:
            raise ValueError("FATAL: GOOGLE_API_KEY and LLM_MODEL must be set.")

        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
        self.web_search_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, google_api_key=google_api_key)
        self.embeddings = get_embedding_model()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))

    def _find_target_graph_top_down(self, query_tag_embedding: list) -> tuple[str, list]:
        """Navigates the tag hierarchy to find the most relevant MetaMedGraph and the path taken."""
        print("U-Retrieval: Starting Top-Down search...")
        path_texts = []

        with self.neo4j_driver.session() as session:
            top_level_res = session.run("""
                MATCH (t:AbstractTag)
                WITH max(t.level) as max_level
                MATCH (top:AbstractTag {level: max_level})
                WITH top, gds.similarity.cosine(top.embedding, $embedding) as score
                RETURN id(top) as nodeId, top.text as text ORDER BY score DESC LIMIT 1
            """, embedding=query_tag_embedding).single()

            if not top_level_res:
                print("Warning: No abstract tag hierarchy found.")
                return None, []

            current_node_id = top_level_res['nodeId']
            path_texts.append(top_level_res['text'])

            while True:
                child_res = session.run("""
                    MATCH (parent)<-[:SUMMARIZES]-(child) WHERE id(parent) = $parent_id
                    WITH child, gds.similarity.cosine(child.embedding, $embedding) as score
                    RETURN id(child) as nodeId, child.text as text, labels(child)[0] as label
                    ORDER BY score DESC LIMIT 1
                """, parent_id=current_node_id, embedding=query_tag_embedding).single()

                if not child_res: break

                current_node_id = child_res['nodeId']
                path_texts.append(child_res['text'])

                if child_res['label'] == 'TagSummary':
                    graph_res = session.run("""
                        MATCH (g:MetaMedGraph)-[:HAS_TAG_SUMMARY]->(t) WHERE id(t) = $node_id
                        RETURN g.chunk_id as chunk_id
                    """, node_id=current_node_id).single()
                    if graph_res: return graph_res['chunk_id'], path_texts
                    else: break
        return None, []

    def _get_initial_context(self, chunk_id: str, query_embedding: list) -> str:
        """Gets the most relevant entities from the target graph and their triple-linked neighbors."""
        print(f"U-Retrieval: Getting initial context from chunk {chunk_id} with limit {self.num_initial_entities}")
        with self.neo4j_driver.session() as session:
            # limit_value = self.initial_context_limit # Old way
            query_text = f"""
                MATCH (g:MetaMedGraph {{chunk_id: $chunk_id}})<-[:PART_OF]-(e:RAG_Entity)
                WITH e, gds.similarity.cosine(e.embedding, $embedding) as score
                ORDER BY score DESC LIMIT {self.num_initial_entities}
                OPTIONAL MATCH (e)-[:IS_REFERENCED_BY]->(p:Paper_Entity)
                OPTIONAL MATCH (e)-[:HAS_DEFINITION_IN]->(u:UMLS_Entity)
                RETURN e.name as name, e.context as context, p.context as paper, u.definition as definition
            """
            # Note: Neo4j parameters $chunk_id and $embedding are still handled by session.run
            result = session.run(query_text, chunk_id=chunk_id, embedding=query_embedding)

            context_parts = []
            for record in result:
                part = f"Entity: {record['name']}. Context: {record['context']}"
                if record['paper']: part += f" | Reference: {record['paper']}"
                if record['definition']: part += f" | Definition: {record['definition']}"
                context_parts.append(part)
            return "\n".join(context_parts)

    def _refine_response_bottom_up(self, response: str, path: list, query: str) -> str:
        """Iteratively refines the answer by moving up the tag hierarchy path."""
        print("U-Retrieval: Starting Bottom-Up refinement...")
        current_response = response
        for summary in reversed(path):
            prompt = ChatPromptTemplate.from_template(
                "Adjust the 'Last Response' using the 'Updated Information'.\n\n"
                "Question: {question}\n"
                "Last Response: {last_response}\n"
                "Updated Information (Topic Summary): {summary}\n\n"
                "Adjusted Response:"
            )
            chain = prompt | self.llm
            current_response = chain.invoke({
                "question": query, "last_response": current_response, "summary": summary
            }).content
        return current_response

    def u_retrieval_node(self, state: GraphState) -> dict:
        print("---NODE: u_retrieval_node---")
        query = state["query"]

        medical_tag_categories_for_prompt = [
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
        categories_str = ", ".join([f"'{cat}'" for cat in medical_tag_categories_for_prompt])

        query_tag_prompt = f"""Your task is to transform a user's medical query into a concise search phrase or a list of 2-3 key medical terms. This output will be used for semantic matching against a knowledge base whose content summaries are structured using the following medical categories: {categories_str}.

Focus on extracting the core medical subject(s) from the user's query, such as the primary medical condition, treatment, diagnostic concept, symptom, or other relevant medical entity. The terms you generate should align well with the kind of information typically found under the aforementioned categories.

User Query: "{query}"

Concise Medical Subject Phrase/Keywords (optimized for matching against the categories above):"""
        query_tag = self.llm.invoke(query_tag_prompt).content

        query_embedding = self.embeddings.embed_query(query_tag)

        target_chunk_id, path = self._find_target_graph_top_down(query_embedding)

        if not target_chunk_id:
            return {"final_answer": "KNOWLEDGE_GAP", "disclaimer_needed": False}

        initial_context = self._get_initial_context(target_chunk_id, query_embedding)
        if not initial_context:
            return {"final_answer": "KNOWLEDGE_GAP", "disclaimer_needed": False}

        prompt_template_str = """You are a highly diligent medical AI assistant. Your primary goal is to accurately answer the user's medical question based *exclusively* on the information contained within the 'Provided Graph Context'.

**Instructions:**
1.  **Strict Grounding:** Base your answer SOLELY on the 'Provided Graph Context'. Do NOT use any external knowledge or make assumptions beyond what is explicitly stated in the context.
2.  **Comprehensive Synthesis:** If multiple pieces of information within the context are relevant to the question, synthesize them into a coherent answer.
3.  **Direct Answer:** Directly answer the question. If the context does not contain the necessary information, clearly and politely state that the answer cannot be found in the provided context.
4.  **Clarity and Precision:** Formulate your answer with clarity and precision, using neutral and objective language appropriate for medical information.

Provided Graph Context:
---
{context}
---
User's Medical Question: {question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        chain = prompt | self.llm
        initial_response = chain.invoke({"context": initial_context, "question": query}).content

        final_answer = self._refine_response_bottom_up(initial_response, path, query)

        return {"final_answer": final_answer, "disclaimer_needed": False}

    def safety_filter_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            """Classify the query as "safe" or "unsafe". Unsafe queries ask for harmful advice.
            Query: {query}"""
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["query"]})
        return {"query_type": "unsafe" if "unsafe" in result.lower() else None}

    def classify_query_node(self, state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            """Classify the query as "foundational", "protocol", or "hybrid".
            Query: {query}"""
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["query"]}).lower().strip()
        if result not in ["foundational", "protocol", "hybrid"]:
            result = "hybrid"
        return {"query_type": result}

    def foundational_node(self, state: GraphState):
        answer = self.llm.invoke(state["query"])
        return {"final_answer": answer.content, "disclaimer_needed": False}

    def hybrid_rag_node(self, state: GraphState):
        query = state["query"]
        enhanced_query_for_web_search = f"""You are an AI assistant tasked with answering a medical question using information obtained from a web search. Your response should be informative and based on the search results, but you must not provide medical advice.

**Instructions:**
1.  **Critical Evaluation:** Critically evaluate the information from the web search results.
2.  **Source Awareness:** If possible, try to synthesize information from multiple reputable sources found in the search. If information comes from a single source or if there are discrepancies, acknowledge this (e.g., "One source suggests X, while another indicates Y." or "According to [Source Type if identifiable, e.g., a health portal], ...").
3.  **Prioritize Reputable Sources:** Give preference to information from well-known medical institutions, research organizations, or governmental health agencies if such sources are discernible in the search results.
4.  **Handle Conflicting Information:** If search results present conflicting information, present the different viewpoints rather than making a definitive statement on the conflict.
5.  **No Medical Advice:** Summarize the information found. Do NOT provide diagnosis, treatment recommendations, or any form of direct medical advice. The goal is to inform, not to prescribe or guide treatment.
6.  **Direct Answer to Query:** Focus on directly answering the user's original question based on the synthesized information.

User's original question: "{query}"

Informative Summary based on Web Search (NOT medical advice):"""
        native_google_search = GenAITool(google_search={}) # GenAITool is imported at the top
        answer_obj = self.web_search_llm.invoke(enhanced_query_for_web_search, tools=[native_google_search])
        final_answer = answer_obj.content
        return {"final_answer": final_answer, "disclaimer_needed": True}

    def canned_safety_response_node(self, state: GraphState):
        return {"final_answer": "I cannot answer this question as it seeks potentially harmful advice. Please consult with a qualified healthcare professional."}

    def log_and_reroute_node(self, state: GraphState):
        print(f"Knowledge Gap found for query: {state['query']}. Rerouting to hybrid.")
        return {}

    def add_disclaimer_node(self, state: GraphState):
        if state.get("disclaimer_needed"):
            disclaimer = ("**Disclaimer:** This information may be based on web search results and is for informational purposes only. It is not a substitute for professional medical advice.")
            return {"final_answer": f"{disclaimer}\n\n---\n\n{state['final_answer']}"}
        return {}

    def decide_after_safety(self, state: GraphState):
        return "canned_safety_response" if state.get("query_type") == "unsafe" else "classify_query"
    def decide_after_classification(self, state: GraphState):
        return state.get("query_type")
    def decide_after_protocol_rag(self, state: GraphState):
        return "log_and_reroute" if "KNOWLEDGE_GAP" in state.get("final_answer", "") else "add_disclaimer"

    def compile_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("safety_filter", self.safety_filter_node)
        graph.add_node("classify_query", self.classify_query_node)
        graph.add_node("foundational", self.foundational_node)
        graph.add_node("u_retrieval", self.u_retrieval_node)
        graph.add_node("hybrid", self.hybrid_rag_node)
        graph.add_node("log_and_reroute", self.log_and_reroute_node)
        graph.add_node("add_disclaimer", self.add_disclaimer_node)
        graph.add_node("canned_safety_response", self.canned_safety_response_node)
        graph.set_entry_point("safety_filter")
        graph.add_conditional_edges("safety_filter", self.decide_after_safety)
        graph.add_conditional_edges("classify_query", self.decide_after_classification)
        graph.add_conditional_edges("u_retrieval", self.decide_after_protocol_rag)
        graph.add_edge("log_and_reroute", "hybrid")
        graph.add_edge("foundational", "add_disclaimer")
        graph.add_edge("hybrid", "add_disclaimer")
        graph.add_edge("add_disclaimer", END)
        graph.add_edge("canned_safety_response", END)
        return graph.compile()
