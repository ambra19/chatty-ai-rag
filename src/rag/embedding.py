from docling.document_converter import DocumentConverter
import os
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.chunking import HybridChunker
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import lancedb
# from typing import List, Any

# # load env
load_dotenv(dotenv_path=".env") 

# Setup LanceDB
db = lancedb.connect("src/lancedb")  # Database path
func = get_registry().get("openai").create(name="text-embedding-3-large")
# ndims = func.ndims()  # Compute dimensions outside the class

# Define metadata schema
class ChunkMetadata(LanceModel):
    filename: str | None
    category: str | None
    chunk_index: int | None

# Define table schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    filename: str | None = None
    category: str | None = None  
    chunk_index: int | None = None

# Create or overwrite table
table = db.create_table("docling", schema=Chunks, mode="overwrite")

# start getting the data and transform it + categories
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

# start tokenizing the data

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

    # Wrap each chunk and add chunk_index
    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "text": chunk.text,
            "filename": source,
            "category": category,
            "chunk_index": idx
        }
        all_chunks.append(chunk_data)

# for chunk in all_chunks:
#     print(f"\n--- Chunk from {chunk.metadata['filename']}, category: {chunk.metadata['category']}, at index: {chunk.metadata['chunk_index']} ---")
#     print({chunk.page_content})

# create a chroma db with all the embeddings
table.add(all_chunks)

# print(f"Added {len(all_chunks)} chunks to LanceDB table '{table.name}'")
# print(f"Table now has {table.count_rows()} rows")

