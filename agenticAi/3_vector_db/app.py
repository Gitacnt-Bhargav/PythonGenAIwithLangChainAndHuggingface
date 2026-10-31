import chromadb

client = chromadb.Client()

collection = client.create_collection("news_articles3")

collection.add(
    ids=["id1", "id2", "id3", "id4"],
    documents=[
        "Apple is leading in a smart phone game with iPhone sale up by 35%",
        "Tesla booked a minor profit of 1 billion $ in Q2",
        "Apples are high in fiber, vitamin C, and various antioxidants",
        "SpaceX got NASA contract worth 10 billion $"
    ]
)

# By default, ChromaDB uses an embedding model for text data.
# Here we provide explicit embeddings so the example runs without extra runtime dependencies.

print(collection)
