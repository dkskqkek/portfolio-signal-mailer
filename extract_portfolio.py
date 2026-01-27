import PyPDF2
import sys


def extract_text(pdf_path):
    print(f"Reading {pdf_path}...")
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            print("--- CONTENT START ---")
            print(text)
            print("--- CONTENT END ---")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    extract_text(r"d:\gg\portfolio_summary.pdf")
