#!/usr/bin/env python
# coding: utf-8

from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os

# ✅ Load the environment variables if needed (optional)
from dotenv import load_dotenv
load_dotenv()

# ✅ Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Define paths for CSV data
data_files = {
    "dil": "Data/dil_scraped_data.csv",
    "isss": "Data/isss_scraped_data.csv",
    "csee": "Data/research_data.csv"
}

# ✅ Load and merge CSV data
dfs = []
for name, file in data_files.items():
    if os.path.exists(file):
        dfs.append(pd.read_csv(file))
    else:
        print(f"⚠️ Warning: {file} not found, skipping.")

if not dfs:
    raise FileNotFoundError("❌ No valid CSV files found. Ensure your data files exist.")

# ✅ Merge datasets and clean missing values
data = pd.concat(dfs[:-1], ignore_index=True)  # Exclude "csee" for now
data["Text"] = data["Text"].fillna("").astype(str)

# ✅ Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# ✅ Initialize ChromaDB
persist_directory = "chroma_store"
chroma_store = Chroma(
    collection_name="retriever_bot",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

# ✅ Process each row in the dataset
for idx, row in data.iterrows():
    if not all([row.get("Section"), row.get("Link"), row.get("Title")]):
        print(f"⚠️ Skipping row {idx} due to missing metadata.")
        continue

    chunks = [chunk for chunk in text_splitter.split_text(row["Text"]) if chunk.strip()]
    if not chunks:
        print(f"⚠️ No valid chunks for row {idx}, skipping.")
        continue

    metadata = {"section": row["Section"], "link": row["Link"], "title": row["Title"]}
    
    try:
        chroma_store.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))
    except Exception as e:
        print(f"❌ Error adding texts for row {idx}: {e}")

# ✅ Process "csee" research data separately
if "csee" in data_files:
    df_csee = pd.read_csv(data_files["csee"])

    persist_directory_research = "chroma_store1"
    chroma_store1 = Chroma(
        collection_name="research_info",
        embedding_function=embedding_model,
        persist_directory=persist_directory_research
    )

    for idx, row in df_csee.iterrows():
        metadata = {"section": row.get("Section", ""), "link": row.get("Link", ""), "title": row.get("Title", "")}
        
        try:
            chroma_store1.add_texts(texts=[row["Text"]], metadatas=[metadata])
        except Exception as e:
            print(f"❌ Error adding research texts for row {idx}: {e}")

print("✅ Data processing completed successfully.")
