"""
Fetches custom document dataset
"""
from dotenv import load_dotenv
load_dotenv()

from optimization_playground_shared.apis.url_to_text import get_document_dataset
import asyncio
import zipfile
import io
import hashlib

async def main():
    zip_buffer = io.BytesIO()
    """
    TODO: add language detection etc to filter out?
    """
    seen_files = set([])
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        async for i in get_document_dataset():
            file_name = hashlib.sha256(i.encode()).hexdigest()
            if file_name in seen_files:
                continue
            seen_files.add(file_name)
            zf.writestr(file_name, i)
            print(file_name, f"size: {zip_buffer.tell() / (10 ** 9)}")

    zip_data = zip_buffer.getvalue()
    with open("text_document_dataset.zip", "wb") as f:
        f.write(zip_data)

if __name__ == "__main__":
    asyncio.run(main())
