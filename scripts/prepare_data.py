# scripts/prepare_data.py
# This script reads PDFs and converts them to clean text files
# Run this ONCE before anything else in the project

import fitz          # pymupdf - reads PDF files
import json          # saves data as JSON format
import os            # handles file paths and folders
import re            # finds and replaces text patterns

def extract_text(pdf_path):
    """
    Opens a PDF and reads every page
    Returns a list of dicts, one per page
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # extract raw text
        pages.append({
            "page": page_num + 1,     # page numbers start at 1
            "text": text,
            "source": pdf_path
        })

    doc.close()
    return pages


def clean_page(text):
    """
    Cleans up messy PDF text
    Removes extra spaces, page numbers, headers
    """
    # collapse 3+ newlines into just 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # collapse multiple spaces into one
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # remove lines that are just a number (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # remove RBI running headers
    text = re.sub(r'Reserve Bank of India\s*\n', '', text)

    return text.strip()


def is_useful_page(text):
    """
    Decides if a page has real content worth keeping
    Skips cover pages, blank pages, broken table pages
    """
    # skip pages with less than 100 characters
    if len(text.strip()) < 100:
        return False

    # skip pages that are mostly line breaks (broken tables)
    if text.count('\n') / max(len(text), 1) > 0.3:
        return False

    return True


def process_pdf(pdf_path, doc_type, year):
    """
    Full pipeline for one PDF:
    1. Extract text from all pages
    2. Clean each page
    3. Filter out useless pages
    4. Return clean pages with metadata
    """
    print(f"Processing: {pdf_path}")

    raw_pages = extract_text(pdf_path)
    clean_pages = []

    for page_data in raw_pages:
        cleaned = clean_page(page_data["text"])

        if is_useful_page(cleaned):
            clean_pages.append({
                "page":     page_data["page"],
                "text":     cleaned,
                "source":   pdf_path,
                "doc_type": doc_type,
                "year":     year
            })

    print(f"  → kept {len(clean_pages)} of {len(raw_pages)} pages")
    return clean_pages


def save_jsonl(pages, output_path):
    """
    Saves clean pages to a .jsonl file
    Each line in the file = one page as JSON
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + '\n')

    print(f"  → saved to {output_path}")


def sanity_check(processed_dir):
    """
    Checks the processed files look healthy
    Counts total words across all documents
    """
    print("\n── Sanity Check ──")
    total_words = 0
    total_files = 0

    for filename in os.listdir(processed_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(processed_dir, filename)
            file_words = 0
            page_count = 0

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    page = json.loads(line)
                    file_words += len(page['text'].split())
                    page_count += 1

            print(f"  {filename}: {page_count} pages, {file_words:,} words")
            total_words += file_words
            total_files += 1

    print(f"\nTotal: {total_files} files, {total_words:,} words")

    if total_words < 1_500_000:
        print("WARNING: word count seems low — check for parsing failures")
    elif total_words > 3_000_000:
        print("WARNING: word count seems high — check for duplicates")
    else:
        print("Word count looks healthy!")


# ─────────────────────────────────────────
# MAIN — runs when you execute this script
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Define all PDFs to process
    # Format: (pdf_path, doc_type, year)
    documents = [
        # RBI Monetary Policy Reports
        ("data/raw/rbi/monetary_policy_report_2022_apr.pdf", "monetary_policy_report", 2022),
        ("data/raw/rbi/monetary_policy_report_2022_oct.pdf", "monetary_policy_report", 2022),
        ("data/raw/rbi/monetary_policy_report_2023_apr.pdf", "monetary_policy_report", 2023),
        ("data/raw/rbi/monetary_policy_report_2023_oct.pdf", "monetary_policy_report", 2023),
        ("data/raw/rbi/monetary_policy_report_2024_apr.pdf", "monetary_policy_report", 2024),
        ("data/raw/rbi/monetary_policy_report_2024_oct.pdf", "monetary_policy_report", 2024),

        # RBI Financial Stability Reports
        ("data/raw/rbi/financial_stability_report_2022_jun.pdf", "financial_stability_report", 2022),
        ("data/raw/rbi/financial_stability_report_2022_dec.pdf", "financial_stability_report", 2022),
        ("data/raw/rbi/financial_stability_report_2023_jun.pdf", "financial_stability_report", 2023),
        ("data/raw/rbi/financial_stability_report_2023_dec.pdf", "financial_stability_report", 2023),
        ("data/raw/rbi/financial_stability_report_2024_jun.pdf", "financial_stability_report", 2024),
        ("data/raw/rbi/financial_stability_report_2024_dec.pdf", "financial_stability_report", 2024),

        # RBI Annual Reports
        ("data/raw/rbi/annual_report_2022.pdf", "annual_report", 2022),
        ("data/raw/rbi/annual_report_2023.pdf", "annual_report", 2023),
        ("data/raw/rbi/annual_report_2024.pdf", "annual_report", 2024),

        # Economic Survey
        ("data/raw/economic_survey/economic_survey_2022_23.pdf", "economic_survey", 2022),
        ("data/raw/economic_survey/economic_survey_2023_24.pdf", "economic_survey", 2023),
    ]

    # Process each PDF
    for pdf_path, doc_type, year in documents:
        if not os.path.exists(pdf_path):
            print(f"SKIPPING (not downloaded yet): {pdf_path}")
            continue

        # Create output filename
        base_name = os.path.basename(pdf_path).replace('.pdf', '.jsonl')
        output_path = os.path.join("data/processed", base_name)

        # Process and save
        pages = process_pdf(pdf_path, doc_type, year)
        save_jsonl(pages, output_path)

    # Run sanity check on everything processed so far
    sanity_check("data/processed")