import chromadb
client = chromadb.Client()
collection = client.create_collection('news_article')
collection.add(ids=['id1','id2','id3','id4'], documents=['Apple is leading in a smart phone game with iPhone sale up by 35%','Tesla booked a minor profit of 1 billion $ in Q2','Apples are high in fiber, vitamin C, and various antioxidants','SpaceX got NASA contract worth 10 billion $'], embeddings=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,1.0,0.0]])
print('ok')
