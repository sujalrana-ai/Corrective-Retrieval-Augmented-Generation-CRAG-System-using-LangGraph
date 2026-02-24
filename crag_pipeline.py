# =====================================
# 1. SETUP & IMPORTS
# =====================================
from dotenv import load_dotenv
load_dotenv()

from typing import List
from typing_extensions import TypedDict
from pprint import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

# =====================================
# 2. DATA LOADING & VECTOR STORE
# =====================================
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

print("---LOADING DOCUMENTS---")
raw_docs = []
for url in urls:
    raw_docs.extend(WebBaseLoader(url).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
documents = splitter.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings, 
    collection_name="crag-chroma"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =====================================
# 3. LOCAL LLM SETUP (Flan-T5)
# =====================================
print("---LOADING LOCAL LLM---")
hf_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# =====================================
# 4. GRAPH STATE & NODES
# =====================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

def retrieve(state: GraphState):
    print("---NODE: RETRIEVE---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"]}

# Grader Node Logic
def grade_documents(state: GraphState):
    print("---NODE: CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        # Simple keyword relevance check or LLM grading
        if any(word in d.page_content.lower() for word in question.lower().split()):
            filtered_docs.append(d)
    
    return {"documents": filtered_docs, "question": question}

def generate(state: GraphState):
    print("---NODE: GENERATE---")
    prompt = PromptTemplate(
        template="Answer using context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    chain = prompt | llm | StrOutputParser()
    
    context = "\n\n".join(d.page_content for d in state["documents"])
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {"generation": answer}

# =====================================
# 5. BUILD THE CRAG GRAPH
# =====================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# =====================================
# 6. RUN THE SYSTEM
# =====================================
inputs = {"question": "What are the components of an LLM agent?"}
for output in app.stream(inputs):
    for node, state in output.items():
        print(f"\nFinished Node: {node}")
    print("---")

print("\nFINAL RESULT:")
pprint(state.get("generation", "No generation found"))