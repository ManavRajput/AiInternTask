##### PDF Reader and Summarizer #####

## Overview ##

This project is a PDF reader and summarizer that extracts text from PDF files, summarizes the content inside the file, and extracts the main importanat keywords.
It uses multiprocessing for parallel processing, allowing the processing of multiple PDFs concurrently and saving more time and utilizing full resources.
The results gained after the processing of the psdf are stored in a MongoDB database.

## Features

- We can connect to a MongoDB database to store PDF summaries, keywords and metadata.
- Processes PDF files from a specified directory.
- Summarizes text using TextRank algorithm.
- Extracts the top keywords using TF-IDF vectorization.
- Handles PDFs of varying lengths (short, medium, and long).
- Utilizes multiprocessing for efficient batch processing.

## Requirements

Before running the project, ensure you have the following packages installed:

- `pymongo`
- `multiprocessing`
- `PyPDF2`
- `nltk`
- `summa`
- `scikit-learn`
- `numpy`

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
