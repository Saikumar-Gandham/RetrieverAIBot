#!/bin/bash

# Run the embedding script
python3 embeddings.py

# Run the helper files 
python3 intenthandler.py
python3 queryhandler.py
python3 prompts.py
python3 feedback.py

# Run the FastAPI server
python3 main.py