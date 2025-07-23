from docling.document_converter import DocumentConverter
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.chunking import HybridChunker
from pprint import pprint


# load env
load_dotenv(dotenv_path=".env") 

# embedding model and vectorDB

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 1. start getting the data and transform it + categories
source = "src/data"  # dir path
result = []
converter = DocumentConverter()

for filename in os.listdir(source):
    filepath = os.path.join(source, filename)

    # convert document
    converted_docs = converter.convert(filepath)

    # Create metadata manually
    category = filename.split(".")[0]

    result.append({
        "document": converted_docs,
        "category": category,
        "name_of_file": filename
    })

# for item in result:
#     print("Category:", item["category"])
#     print("Source file:", item["name_of_file"])
#     print("Content (markdown):", item["document"].document.export_to_markdown())


# 2. start tokenizing the data

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunker = HybridChunker(
    model="cl100k_base",
    max_tokens=500,
    overlap_tokens=100,
    merge_peers=True
)

all_chunks = []

for item in result:
    docling_doc = item["document"].document
    category = item["category"]
    source = item["name_of_file"]

    # Chunk the document
    chunks = list(chunker.chunk(dl_doc=docling_doc))

    # Wrap each chunk and add chunk_index, makes it
    # easier to show to the user the source
    for idx, chunk in enumerate(chunks):
        all_chunks.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "category": category,
                    "filename": source,
                    "chunk_index": idx
                }
            )
        )

for chunk in all_chunks:
    print(f"\n--- Chunk from {chunk.metadata['filename']}, category: {chunk.metadata['category']}, at index: {chunk.metadata['chunk_index']} ---")
    # print({chunk.page_content})
    