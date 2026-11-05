"""
in Ingestion pipeline, we used - CharacterTextSplitter - it justs cuts text at fixed character counts. Simple, but crude

1. CharacterTextSplitter ( Beyond basic chunk_size)
    > Custom separators (split on specific patterns)
    > Still useful for simple, uniform documents or when speed matters most

2. RecursiveCharacterTextSplitter ( Upgrade from CharacterTextSplitter )
    > Tries to split at natual boundaries (paragraphs, sentences, words)
    > Falls back gracefully if chunk is too big
    > Preserves more context than basic splitting

3. Document-Specific Splitting (Respects document structure)
    > PDF: Splits by pages, sections, headers
    > Markdown: Splits by headers, code blocks, lists
    > Each document type gets appropriate treatment

4. Semantic Splitting (Content-aware boundaries)
    > Uses embeddings to detect topic shifts
    > Keeps related concepts together
    > Splits when meaning changes, not just by size
    > More intelligent but computationally expensive

5. Agentic Splitting (AI-powered chunking)
    > LLM analyzes content and decides optimal splits
    > Can understand complex relationships
    > Adapts to content type automatically
    > Most sophisticated but slowest/most expensive
    

CharacterTextSplitter:

> CharacterTextSplitter doesnt just cut at character limits. It follows a split-first, merge second approach
    > Split: Break text at separators (default: \n\n)
    > Merge: Combine pieces until hitting chunk_size limit
    > When splitting, if a single para itself is more than chunk_size before hitting the separator, then CharacterTextSplitter 
    will keep that para as it is as a chunk even if the size overpasses the chunk_size
    > separator="\n\n" or "\n" or "." or " " or "" (priority decreases from "\n\n" to "")
    > only one separator can be given here

RecursiveCharacterSplitter:

> 


"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

print("\n" + "=" *60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators= ["\n\n","\n",". "," ", ""], 
    # here can give multiple separators and recursively selects each separator and splits until it reaches below chunk_size
    chunk_size = 100,
    chunk_overlap=0
)

chunks2 = recursive_splitter.split_text(tesla_text)
print(f"Same problem text, but with RecursiveCharacterTextSplitter")

for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

