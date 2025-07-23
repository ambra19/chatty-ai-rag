import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=".env") 
api_key = os.getenv("GROQ_API_KEY")

# llm = ChatOpenAI(model="gpt-4o-mini")

llm = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=api_key,
    reasoning_format="raw"
    # reasoning_effort="default"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = InMemoryVectorStore(embeddings)


# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("",), #"https://lilianweng.github.io/posts/2023-06-23-agent/"
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# define user as user input
# user = input("User: ")
response = graph.invoke({"question": "Conform Normativului GE 032-97, care este durata de viata a obiectivului?", "context": [], "answer": ""}) #"Hello, what is SCoRe?"
# response = graph.invoke({"question": "Hello"})
# print(response["answer"])

# for document in response["context"]:
#     print(document)
