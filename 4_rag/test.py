from langchain_huggingface import HuggingFaceEmbeddings
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")    
embeddings = model.embed_query("Your text here")
print(embeddings)
