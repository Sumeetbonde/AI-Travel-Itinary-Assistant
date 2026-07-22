import os

from fastapi import UploadFile
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    get_embeddings,
)


async def ingest_pdf(file: UploadFile):
    print(f"📄 Processing PDF file: {file.filename}")

    content = await file.read()

    # Load PDF directly from memory
    docs = []
    pdf = fitz.open(stream=content, filetype="pdf")

    page_count = len(pdf)
    print(f"📑 PDF loaded successfully. Found {page_count} pages")

    try:
        for page_num in range(page_count):
            page = pdf[page_num]
            text = page.get_text()

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "source": file.filename
                    }
                )
            )

    finally:
        pdf.close()

    print("✂️ Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    
    for i, chunk in enumerate(chunks):
        print(f"\n===== CHUNK {i + 1} =====")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)


    print(f"📚 Created {len(chunks)} text chunks")

    # Optional debug
    # print(chunks)

    print("🧮 Generating embeddings...")

    embedding_function = get_embeddings()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print(f"📝 Prepared {len(texts)} texts for embedding")

    os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

    print(
        f"⬆️ Uploading {len(chunks)} documents to collection "
        f"'{CHROMA_COLLECTION_NAME}'..."
    )

    Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )

    print(
        f"✅ Upload complete! Added {len(chunks)} chunks "
        f"from {page_count} pages"
    )

    return {
        "filename": file.filename,
        "pages": page_count,
        "chunks": len(chunks),
        "collection": CHROMA_COLLECTION_NAME,
        "persist_directory": CHROMA_PERSIST_DIRECTORY,
        "status": "success",
        "chunk_contents": [
            chunk.page_content for chunk in chunks
        ]

    }