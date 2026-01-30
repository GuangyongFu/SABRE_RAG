# -*- coding: utf-8 -*-
"""
SABRE_RAG.py

Local execution setup.

# Large language models (LLMs) and retrieval augmented generation (RAG) 
for physics-specific queries using the OpenAI API.

2025-01-15 - James Verbus
modified on 2025-01-16 by Guangyong Fu

# Prerequisites

1) Install required libraries by running:
   pip install -r requirements.txt

2) Set up OpenAI API keys as environment variables:
   export OPENAI_API_KEY='your_api_key'
   export OPENAI_ORGANIZATION='your_organization_id'
"""

import os
import openai
from llama_index.core import ServiceContext, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI

# Set OpenAI API keys
openai.organization = ''
openai.api_key = ''

# Define queries
questions = {
    1: "What's the crystal threshold of SABRE experiment?",
    2: "How does SABRE South detector work?",
    3: "How are the SABRE South crystals grown?",
    # Add your questions 6, 7, ... here
}

# Query OpenAI GPT-4o-mini directly (without RAG)
client = OpenAI(api_key=openai.api_key)

def query_oai(question):
    chat_completion = client.chat_completions.create(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion

# Setup basic RAG system
local_documents_path = './documents/papers'  # Update to your local path
Settings.chunk_size = 1000
Settings.chunk_overlap = 100
Settings.llm = OpenAI(model="gpt-4o-mini")

# Read local documents with Llama index
documents = SimpleDirectoryReader(local_documents_path, recursive=True).load_data()

# Create index from documents
index = VectorStoreIndex.from_documents(documents)

# Query with RAG
def query_rag(question):
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(question)
    return response

# Run queries
#for q_id, question in questions.items():
#    print(f"## Question {q_id}")
#    response = query_rag(question)
#    print(f"Question: {question}")
#    print(f"Response: {response.response}\n")

    # Display source nodes
#    for node in response.source_nodes:
#        filename = node.metadata.get('file_name', 'Unknown')
#        page_label = node.metadata.get('page_label', 'Unknown')
#        #text = node.text
#        print(f"Page: {page_label}, File: {filename}\n")

