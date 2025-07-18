import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core import embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, sentence_transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
# 1. Load web article
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed and store
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)


from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Use OpenAI chat model
llm = GoogleGenerativeAI(model="gemini-2.0-flash-exp")  

# Build a retrieval-augmented chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

def ask(question: str):
    result = qa_chain(question)
    print("\n📘 Answer:", result["result"])
    print("📚 Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", ""))

if __name__ == "__main__":
    ask("What is the main idea of the article?")
