
"""

SEMANTIC CHUNKER

Semantic chunker breaks up long documents into meaningful pieces by finding where topics natually change.
Instead of cutting at random word counts, it uses AI embeddings to understand the semantic meaning of sentences.

If one sentence talks about a certain topic and the next sentence talks about an entire different topic, then it means
we need to chunk

3 Step process:
1. Encode: Convert each sentence into embeddings (numerical values)
2. Compare: Calculate similarity score between nearby sentences.
3. Split: Create boundaries where similarity drops significantly

Calculate the percentile:
1. Put all scores in order from lowest to highest, for eg. there are list of 10 scores
2. Pick your percentile, eg. 70th percentile
3. Calculate which position to look at - 0.70 * 10 = 7th position
4. Find the score at that position, which is 7th position in our list
70th percentile of all scores means, there are 7 out of 10 score below the 70th percentile value

Once each sentence similartiy score is calculated, they are maintained in a list,
like 1st value in the list would be the similarity score between 1st and 2nd sentence, and so on

Now, each value in list is traversed, if the value is less than the mentioned percentile, like here its 70,
means it will split wherever similartiy is <= 0.71.

for eg. 
similarity score between 1st and 2nd sentence 0.85 -> keep together (above 0.71)
similarity score between 2nd and 3rd sentence 0.78 -> keep together (above 0.71)
similarity score between 3rd and 4th sentence 0.42 -> SPLIT HERE (below 0.71)
similarity score between 4th and 5th sentence 0.71 -> keep together (above 0.71)
similarity score between 5th and 6th sentence 0.95 -> keep together (above 0.71)

Result: Split after sentence 3 (the topic shift)

Why use Percentile:
Problem: Diff docs have diff similarity patterns

Percentile helps in picking some value from that list instead of hardcoding some random 
value which might not be falling in the range of the values in the list. (all values in list
may be above or below that, making it hard to split). 

Percentile range suggested to be between 60th to 90th.
70th percentile is popular because its not too aggresive (wouldnt split at every small dip) nor its too conservative (wouldnt miss obvious topic changes)


However, SemanticChunker is not feasible in prod because of sheer number of docs that enterprises tend to have
and its not economical as well because we have to use the embedding models in order to do all of the processing and its not viable

"""
from dotenv import load_dotenv

load_dotenv()

tesla_text = """Tesla's Q3 results
Tesla reported record revenue of $25.20 in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached on all-time high of 96%.
Model Y now represent 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings



#Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile", # or "standard deviation"
    breakpoint_threshold_amount=70
)

chunks = semantic_splitter.split_text(tesla_text)

print("=== SEMANTIC CHUNKING RESULTS ==========")

print("=" * 50)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f"{chunk}")
    print()



# *****************************************

