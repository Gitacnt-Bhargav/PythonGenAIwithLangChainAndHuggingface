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
query = "What was the original name of Microsoft before it became Microsoft?"

# retriever = db.as_retriever(search_kwargs = {"k":5}) # retriever retrieves top 5 chunks with highest similarity scores to the user's query embedding

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


#Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this quesions: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])} #chr(10) is like new line character


Please provide a clear, helpful answer using only the information from these documents. If you cant find the answer in the documents, say "I don't have enough information to answer that question based on provided documents"

"""

#Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

#Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

#Invoke the model with the combined input
result = model.invoke(messages)

#Display the full result and content only
print("\n---Generated Response ---")

#print("Full result:")
#print(result)
print("Content only")
print(result.content)




# Synthetic questions:

#1. "What was NVIDIA's first graphics accelerator called"
#2. "Which company did NVIDIA acquire to enter the mobile processor market?"
#3. "What was the Microsoft's first software product?"
#4. "How much did Microsoft pay to acquire Github?"
#5. "In which year did Tesla began production of Roadster?"
#6. "Who succeeded Ze'ev Drori as CEO in October 2008"
#7. "What was the name of the autonomous spaceport drone ship that acheived the first successful sea"
#8. "What was the original name of Microsoft before it became Microsoft?"
#9. "Which island does SpaceX lease for its launches in Pacific?"