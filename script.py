
from pymongo import MongoClient
import multiprocessing
import os
import time
import PyPDF2
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from summa import summarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



def connect_to_mongo():
    client = MongoClient("mongodb+srv://<username>:<password>@pdfcluster.9e0sh.mongodb.net/?retryWrites=true&w=majority&appName=<databasename>") #insert your username,password and database name for Mongdb connection
    db = client['pdf_reader_db']
    collection = db['pdf_summaries']
    return collection

def scan_pdf_files(folder_path):
    pdf_files = multiprocessing.JoinableQueue()

    # Scan for PDF files and add them to the queue
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.put(os.path.join(root, file))  # Add file path to the queue

    # Create a new queue for updated paths
    updated_queue = multiprocessing.JoinableQueue()

    # Update each file path in the queue
    while not pdf_files.empty():
        pdf = pdf_files.get()  # Get the next PDF from the queue
        updated_pdf = pdf.replace("\\", '/')  # Replace backslashes with forward slashes
        updated_queue.put(updated_pdf)  # Put the updated path into the new queue
        pdf_files.task_done()  # Mark the task as done

    return updated_queue  # Return the updated queue


def length_of_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)  # Load the PDF file
            num_pages = len(reader.pages)
        return num_pages
    except Exception as e:
        return f"Error: {e}"



def clean_text(text):
    stop_words = set(stopwords.words('english'))
    # Remove punctuation and lowercase the text
    text = text.lower()    #.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a single string
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        total_pages = length_of_pdf(pdf_path)

        # Extract text in chunks of 10 pages
        chunks = []
        chunk_size = 10
        for i in range(0, total_pages, chunk_size):
            chunk_text = ""
            for j in range(i, min(i + chunk_size, total_pages)):
                page = reader.pages[j]
                chunk_text += page.extract_text()  # Extract raw text from each page

            # Clean the chunked text using clean_text function
            cleaned_chunk = clean_text(chunk_text)
            chunks.append(cleaned_chunk)
        return chunks


# Function to extract, clean, and summarize text from PDF using TextRank
def summarize_all_chunks(chunks):
    if len(chunks) == 1:
        big_chuck = chunks[0] # Use the entire text as the summary for one page
    summaries = []  # Initialize an empty list to store summaries
    for chunk_text in chunks:
        # Summarize 20% of each cleaned chunk text using TextRank
        summary = summarizer.summarize(chunk_text, ratio=0.2)
        summaries.append(summary)  # Append each summary to the list

    # Combine all summaries into one "Big Chunk"
    big_chunk = ' '.join(summaries)
    return big_chunk  # Return the combined summary


# Function to summarize the "Big Chunk"
def summarize_big_chunk(big_chunk, max_length = 150):
    final_summary = summarizer.summarize(big_chunk, ratio=0.2)  # Summarize 20% of the big chunk
    final_summary_words = final_summary.split()
    if len(final_summary_words) > max_length:
        final_summary = ' '.join(final_summary_words[:max_length])

    final_summary = final_summary.lower().translate(str.maketrans('', '', string.punctuation + string.digits))
    return final_summary





def extract_top_keywords_from_chunks(text_chunks, top_n=10):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_chunks)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Get the sum of the TF-IDF scores for each term
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

    # Create a list of (word, score) tuples
    word_scores = [(word, score) for word, score in zip(feature_names, tfidf_scores)]

    # Sort by score (higher score comes first) and get the top N keywords
    top_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]

    # Return only the keywords (words)
    return [keyword for keyword, score in top_keywords]


def process_pdf(pdf_path, collection):
    total_start_time = time.time()
    num_pages = length_of_pdf(pdf_path)
    print("number of pages : \n", num_pages)
    # Generate summary and extract keywords
    chunks = extract_text_from_pdf(pdf_path)  # Extract and clean the text
    big_chunk = summarize_all_chunks(chunks)  # Summarize all cleaned chunks into "Big Chunk"
    final_summary = summarize_big_chunk(big_chunk)  #find the top 10 keywords using nltk
    keywords = extract_top_keywords_from_chunks(chunks)
    # Store the document summary and keywords in MongoDB
    document = {
        "filename": os.path.basename(pdf_path),
        "summary": final_summary,
        "keywords": keywords,
        "num_pages": num_pages
    }
    collection.insert_one(document) #inserting the processed in fo into the server
    total_end_time = time.time()  # calculating the time for processing one pdf
    total_processing_time = total_end_time - total_start_time
    print(f"Processing done for {os.path.basename(pdf_path)}\n")
    print(f"Total time to process {os.path.basename(pdf_path)}: {total_processing_time:.2f} seconds \n")



def worker(queue):
    collection = connect_to_mongo()
    while True:
        pdf_file = queue.get()  # Get the next PDF from the queue
        if pdf_file is None:  # Exit condition
            break
        try:
            process_pdf(pdf_file, collection)
        except Exception as e:
            print(f"Error processing file: {e}")
        finally:
            queue.task_done()  # Mark the task as done  # Ensure task_done is called even if there's an error


def ingest_pdfs(folder_path):
    pdf_files = scan_pdf_files(folder_path)  # This still works with os.walk

    num_workers = 2  # Set the number of worker processes You can change this as per your computational power
    processes = []

    # Start multiple worker processes
    for _ in range(num_workers):
        process = multiprocessing.Process(target=worker, args=(pdf_files,))
        process.start()
        processes.append(process)

    pdf_files.join()  # This blocks until all tasks are done

    # Send exit signals to workers
    for _ in range(num_workers):
        pdf_files.put(None)  # Send a signal to each worker to exit

    # Ensure all processes are done
    for process in processes:
        process.join()

    # Main entry
if __name__ == "__main__":
    path = "c:/user/file_name" #specify your folder path where pdf files are stored
    folder_path = path
    ingest_pdfs(folder_path)
    print("......................................EXECUTION COMPLETED..................................................")

