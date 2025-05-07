from optimization_playground_shared.apis.openai import OpenAiCompletion
from optimization_playground_shared.apis.url_to_text import get_text
import argparse
import fitz
import base64
import requests
import os
from io import BytesIO
from PIL import Image
import hashlib

def download_pdf(url, cache_path="cached_pdf.pdf"):
    if os.path.exists(cache_path):
        print(f"Using cached PDF at {cache_path}")
        return cache_path    
    print(f"Downloading PDF from {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(cache_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF cached at {cache_path}")
    return cache_path

def pdf_to_base64(pdf_path):
    doc = fitz.open(pdf_path)
    base64_images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = pix.tobytes("png")
        
        img_base64 = base64.b64encode(img).decode('utf-8')
        base64_images.append(img_base64)
    
    doc.close()
    return base64_images


def process_pdf(pdf_url):
    cache_path = hashlib.sha256(pdf_url.encode()).hexdigest()
    os.makedirs(".cache", exist_ok=True)
    cache_path = f".cache/{cache_path}.pdf"
    pdf_path = download_pdf(pdf_url, cache_path)
    return pdf_to_base64(pdf_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give an url and get the summary")
    parser.add_argument("url", type=str, help="The url argument")
    args = parser.parse_args()

#    api = OpenAiCompletion(model="gpt-4.1-mini")
    api = OpenAiCompletion(model="gpt-4.1")
    pages = process_pdf(args.url)
    for i in pages:
        print(api.process_image(
            """
            If we are in on the list of references page, then give me a JSON version of those references.
            
            For instance if you see something like
            [1] Elvira Albert, Pablo Gordillo, Benjamin Livshits, Albert Rubio, and Ilya
                Sergey. Ethir: A framework for high-level analysis of ethereum bytecode.
                In Automated Technology for Verification and Analysis (ATVA). Springer,
                2018.

            Return 
            {
                "title": "Ethir: A framework for high-level analysis of ethereum bytecode.",
                "authors": "Elvira Albert, Pablo Gordillo, Benjamin Livshits, Albert Rubio, and Ilya Sergey",
                "publication": "In Automated Technology for Verification and Analysis (ATVA), Springer, 2018."
            }

            Please split out urls whenever possible so that it's easier to iterate over them. Note that there might be none citations on the page, please just return [] in that case. PLease return nothing except references.
            """,
            i
        ))
