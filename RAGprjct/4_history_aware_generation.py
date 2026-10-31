"""
The key difference between basic RAG and history aware RAG is:

In basic RAG, each query is treated independently. The retriever taks your exact question and searches for chunks.

In history-aware RAG, there's one crucial extra step: query reformulation.

Before searching, the system looks at the conversation history and rewrites vague or context-dependent questions into clear, standalone questions
This is for follow-up questions
"""

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chromadb"

#Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

model = ChatOpenAI(model = "gpt-4o")

#Store the conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n Your asked: {user_question} ---")

    #Step 1: Make the question clear using conversation history
    if chat_history:
        #Ask AI to make question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the question as standalone and searchable. Just return the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")

    else:
        search_question = user_question


    #Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs = {"k":3})
    docs = retriever.invoke(search_question)


    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        #Show first 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f" Doc {i} : {preview}... ")


    #Step 3: Create final prompt
    combined_input = f"""Based on following documents, please answer this question: {user_question} 

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you cant find the answer in the documents, say "I don't have enough information to answer that question based on provided documents"

    """

    #Step 4: Get the answer
    messages = [
        SystemMessage(content = "You are a helpful assistant that answers based on provided documents and conversation history"),
        HumanMessage(content=combined_input)
    ]

    #Invoke the model with the combined input
    result = model.invoke(messages)

    #Display the full result and content only
    print("\n --- Generated Response ---")
    #print("Full result: ")
    #print(result)

    print("Content only")
    print(result.content)

    answer = result.content

    #Step5 5: Remember the conversation
    chat_history.append(HumanMessage(content = user_question))
    chat_history.append(AIMessage(content = answer))

    print(f"Answer: {answer}")
    return answer

#Simple chat loop
def start_chat():
    print("Ask me question!. Type 'quit' to exit")

    while True:
        question = input("\n Your question: ")

        if question.lower() == 'quit':
            print("Goodbye!!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()