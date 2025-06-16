# src/tools.py

from langchain.tools import Tool
from Bio import Entrez
import os
import sys
from urllib.error import HTTPError

entrez_email = os.getenv("ENTREZ_EMAIL", "default.user@example.com")
entrez_api_key = os.getenv("ENTREZ_API_KEY")

Entrez.email = entrez_email
if entrez_api_key:
    Entrez.api_key = entrez_api_key

def search_pubmed(query: str) -> str:
    """
    Searches PubMed for a given query and returns a summary of the top 3 results' abstracts.
    """
    try:
        print(f"---TOOL: Searching PubMed for: {query}---")
        handle = Entrez.esearch(db="pubmed", term=query, retmax="3", sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        if not id_list:
            return "No relevant articles found on PubMed for that query."
        
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts = handle.read()
        handle.close()
        return abstracts
    except Exception as e:
        print(f"ERROR in PubMed search: {e}", file=sys.stderr)
        return f"An error occurred while searching PubMed: {e}"

pubmed_tool = Tool(
    name="PubMedSearch",
    func=search_pubmed,
    description="Use this tool to find specific scientific or medical research papers from the PubMed database."
)