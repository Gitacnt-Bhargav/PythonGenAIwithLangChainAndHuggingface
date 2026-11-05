from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

persistent_directory = "db/chromadb"

#Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#search for relevant documents
query = "In which year did Tesla began production of Roadster?"

# retriever = db.as_retriever(search_kwargs = {"k":5}) # retriever retrieves top 3 chunks with highest similarity scores to the user's query embedding

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":5,
        "score_threshold": 0.3 # Only returns chunks with cosine similarity >=0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

#display results
print("---Context---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")



# Synthetic questions:

#1. "What was NVIDIA's first graphics accelerator called"
#2. "Which company did NVIDIA acquire to enter the mobile processor market?"
#3. "What was the Microsoft's first hardware product?"
#4. "How much did Microsoft pay to acquire Github?"
#5. "In which year did Tesla began production of Roadster?"
#6. "Who succeeded Ze'ev Drori as CEO in October 2008"
#7. "What was the name of the autonomous spaceport drone ship that acheived the first successful sea"
#8. "What was the original name of Microsoft before it became Microsoft?"
#9. "Which island does SpaceX lease for its launches in Pacific?"