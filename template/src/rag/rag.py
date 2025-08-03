import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import lancedb
from lancedb.embeddings import get_registry
import os

# Load environment variables
load_dotenv()
client = OpenAI()

DB_DIR = "src/lancedb"  # Updated path to match your embedding script
TABLE_NAME = "docling"

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize LanceDB connection."""
    db = lancedb.connect(DB_DIR)
    
    # Check if table exists
    try:
        table = db.open_table(TABLE_NAME)
        
        # Set up the embedding function for search
        func = get_registry().get("openai").create(name="text-embedding-3-large")
        table = table.create_index("vector", config=lancedb.index.IvfPq())
        
        return table, func
    except Exception as e:
        st.error(f"Could not open LanceDB table '{TABLE_NAME}'. Make sure you've run the embedding script first. Error: {e}")
        return None, None

def get_context(query: str, table, func, num_results: int = 3) -> str:
    """Search the LanceDB table for relevant context."""
    if table is None or func is None:
        return "No database connection available."
    
    try:
        # Alternative approach: manually generate embedding and search
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_vector = embeddings.embed_query(query)
        
        # Search using the query vector
        results = table.search(query_vector, vector_column_name="vector").limit(num_results).to_list()
        contexts = []

        for result in results:
            # Extract content and metadata from LanceDB results
            text = result.get("text", "")
            filename = result.get("filename", "Unknown file")
            category = result.get("category", "Unknown category")
            chunk_index = result.get("chunk_index", "N/A")
            
            # Format similar to original
            source = f"\nSource: {filename} (chunk {chunk_index})"
            title = f"Category: {category}"
            
            contexts.append(f"{text}{source}\nTitle: {title}")

        return "\n\n".join(contexts)
    
    except Exception as e:
        st.error(f"Error searching database: {e}")
        return "Error retrieving context from database."

def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API with retrieved context."""
    system_prompt = f"""You are a helpful assistant that answers questions based only on the provided context.
If the context does not contain enough information, reply that you don't know. Always greet the user back if they greet you.

Context:
{context}
"""

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    # Streaming response from OpenAI
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
    )

    response = st.write_stream(stream)
    return response

# Initialize Streamlit app
st.title("📚 Document Q&A (LanceDB RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table, func = init_db()

# Show database status
if table is not None:
    try:
        row_count = table.count_rows()
        st.success(f"✅ Connected to LanceDB - {row_count} chunks available")
    except:
        st.warning("⚠️ Connected to LanceDB but couldn't get row count")
else:
    st.error("❌ No database connection. Please run the embedding script first.")
    st.stop()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve context
    with st.status("🔎 Searching relevant chunks...", expanded=False):
        context = get_context(prompt, table, func)

        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                # background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                # color: black !important;
                font-weight: 500;
            }
            .search-result summary:hover {
                # color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                # color: #000000;
                font-style: italic;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.write("### 📄 Retrieved Context:")
        
        # Handle case where no context is retrieved
        if context and context != "No database connection available." and context != "Error retrieving context from database.":
            for chunk in context.split("\n\n"):
                if chunk.strip():  # Only process non-empty chunks
                    parts = chunk.split("\n")
                    text = parts[0] if parts else ""
                    metadata = {}
                    
                    # Parse metadata from remaining parts
                    for line in parts[1:]:
                        if ": " in line:
                            key, value = line.split(": ", 1)
                            metadata[key] = value

                    source = metadata.get("Source", "Unknown source")
                    title = metadata.get("Title", "Untitled section")

                    st.markdown(
                        f"""
                        <div class="search-result">
                            <details>
                                <summary>{source}</summary>
                                <div class="metadata">{title}</div>
                                <div style="margin-top: 8px;">{text}</div>
                            </details>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("No relevant context found or database error occurred.")

    # Get assistant response
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context)

    st.session_state.messages.append({"role": "assistant", "content": response})