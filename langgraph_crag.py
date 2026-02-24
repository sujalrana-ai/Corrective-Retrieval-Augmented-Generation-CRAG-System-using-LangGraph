# =====================================
# 1. BASIC SETUP
# =====================================
from dotenv import load_dotenv
load_dotenv()

from typing import List
from typing_extensions import TypedDict
from pprint import pprint

# =====================================
# 2. LANGCHAIN IMPORTS
# =====================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================
# 3. HUGGING FACE LOCAL LLM (STABLE)
# =====================================
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =====================================
# 4. LANGGRAPH
# =====================================
from langgraph.graph import StateGraph, END

# =====================================
# 5. LOAD KNOWLEDGE BASE
# =====================================
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

raw_docs = []
for url in urls:
    raw_docs.extend(WebBaseLoader(url).load())

# =====================================
# 6. SPLIT DOCUMENTS
# =====================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0
)
documents = splitter.split_documents(raw_docs)

# =====================================
# 7. VECTOR STORE (CHROMA)
# =====================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="rag-chroma"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =====================================
# 8. LOAD HF MODEL ONLY ONCE
# =====================================
hf_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# =====================================
# 9. GRAPH STATE
# =====================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

# =====================================
# 10. RETRIEVE NODE (FIXED)
# =====================================
def retrieve(state: GraphState):
    print("---RETRIEVE---")
    docs = retriever.invoke(state["question"])
    return {
        "documents": docs,
        "question": state["question"]
    }

# =====================================
# 11. GENERATE NODE
# =====================================
prompt = PromptTemplate(
    template="""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

rag_chain = prompt | llm | StrOutputParser()

def generate(state: GraphState):
    print("---GENERATE---")
    answer = rag_chain.invoke({
        "context": format_docs(state["documents"]),
        "question": state["question"]
    })
    return {"generation": answer}

# =====================================
# 12. BUILD LANGGRAPH
# =====================================
graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# =====================================
# 13. RUN THE GRAPH (CORRECT WAY)
# =====================================
from pprint import pprint

inputs = {"question": "Explain short term memory and long term memory???"}

final_state = None

for output in app.stream(inputs):
    for node, state in output.items():
        pprint(f"Node '{node}'")
        final_state = state
    pprint("\n---\n")

print("\nFINAL ANSWER:\n")
pprint(final_state["generation"])
