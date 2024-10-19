
from pdfminer.high_level import extract_text
from transformers import pipeline
from keybert import KeyBERT
from pymongo import MongoClient
import queue
import PyPDF2
import multiprocessing
import os
import time
from queue import Queue
from multiprocessing import Queue


def connect_to_mongo():
    client = MongoClient("mongodb+srv://Rose:Manav@pdfcluster.9e0sh.mongodb.net/?retryWrites=true&w=majority&appName=pdfcluster")
    db = client['pdf_reader_db']
    collection = db['pdf_summaries']
    return collection

def scan_pdf_files(folder_path):
    pdf_files = multiprocessing.JoinableQueue()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.put(os.path.join(root, file))


    # Assuming pdf_queue is already populated with file paths
    updated_queue = multiprocessing.JoinableQueue()  # Create a new queue for updated paths

    # Update each file path in the queue
    while not pdf_files.empty():
        pdf = pdf_files.get()  # Get the next PDF from the queue
        updated_pdf = pdf.replace("\\", '/')  # Replace backslashes with forward slashes
        updated_queue.put(updated_pdf)  # Put the updated path into the new queue

    # If you want to replace the original queue with the updated one
    pdf_files = updated_queue
    return pdf_files

def length_of_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)  # Load the PDF file
            num_pages = len(reader.pages)
            print("number of pages : " ,num_pages)
        return num_pages
    except Exception as e:
        return f"Error: {e}"


def read_pdf(pdf_path, max_pages=5):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)

        # Process only the first `max_pages` or the total pages, whichever is smaller
        for page_num in range(min(max_pages, total_pages)):
            text += reader.pages[page_num].extract_text() + "\n"

    return text


def  generate_summary(txt, pdf_length):
    summarizer = pipeline('summarization', model = 't5-base')
    if pdf_length <= 10:
        max_length = 50
    elif pdf_length <= 30:
        max_length = 100
    else:
        max_length = 150

    summary = summarizer(txt, max_length=max_length, min_length = 30, do_sample = False)
    return summary[0]['summary_text']


def extract_keywords(text):
    kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
    return [keyword[0] for keyword in keywords]


def process_pdf(pdf_path, collection):
    total_start_time = time.time()
    text = read_pdf(pdf_path)
    num_pages = length_of_pdf(pdf_path)

    # Generate summary and extract keywords
    summary = generate_summary(text, num_pages)
    keywords = extract_keywords(text)
    # Store the document summary and keywords in MongoDB
    document = {
        "filename": os.path.basename(pdf_path),
        "summary": summary,
        "keywords": keywords,
        "num_pages": num_pages
    }
    collection.insert_one(document)

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    print(f"Total time to process {os.path.basename(pdf_path)}: {total_processing_time:.2f} seconds")
    print("Processing done")


def worker(queue):
    collection = connect_to_mongo()
    while True:
        try:
            pdf_file = queue.get()
            print(pdf_file ,"   : is being processed.....")
            process_pdf(pdf_file, collection)
            queue.task_done()
        except Exception as e:
            print(f"Error processing file: {e}")
            queue.task_done()  # Ensure task_done is called even if there's an error


def ingest_pdfs(folder_path):
    pdf_files = scan_pdf_files(folder_path)  # This still works with os.walk

    num_workers = 2  # Set the number of worker processes
    processes = []

    # Start multiple worker processes
    for _ in range(num_workers):
        process = multiprocessing.Process(target=worker, args=(pdf_files,))
        process.start()
        processes.append(process)

    # Wait for the queue to be processed
    pdf_files.join()

    # Ensure all processes are done
    for process in processes:
        process.join()

# Main entry
if __name__ == "__main__":
    path = "C:/Users/manu/PycharmProjects/pythonProject8/data"
    folder_path = path
    ingest_pdfs(folder_path)
    # pdf_queue = scan_pdf_files(folder_path)
    # pdf_queue = multiprocessing.JoinableQueue()  # Use multiprocessing Queue
    # pdf_files = scan_pdf_files(folder_path)  # This still works with os.walk
    #
    # # Add all the scanned PDFs to the multiprocessing queue
    # for pdf in pdf_files:  # Add each PDF file path to the queue
    #     pdf_queue.put(pdf)
    #
    # while not pdf_queue.empty():  # Continue while the queue is not empty
    #      pdf = pdf_queue.get()  # Get the next PDF from the queue
    #      print(pdf)
    # process_pdf("C:/Users/manu/PycharmProjects/pythonProject8/data/1.pdf", connect_to_mongo())
