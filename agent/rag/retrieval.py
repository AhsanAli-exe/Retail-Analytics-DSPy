import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DOCS_DIR = os.path.join(PROJECT_ROOT,"docs")

class DocumentChunk:
    "Represents a single chunk of a document"
    
    def __init__(self,chunk_id,content,source_file):
        self.chunk_id = chunk_id
        self.content = content
        self.source_file = source_file
        self.score = 0.0
        
def load_documents():
    docs = []
    print(f"DEBUG: Looking in folder: {DOCS_DIR}")
    print(f"DEBUG: Files found: {os.listdir(DOCS_DIR)}")
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".md"):
            file_path = os.path.join(DOCS_DIR,file)
            with open(file_path,"r",encoding="utf-8") as f:
                content = f.read()
            docs.append({
                "filename":file,
                "content":content
            })
    return docs

def chunk_doc(file,content):
    chunks = []
    base_name = file.replace(".md","")
    if"\n## " in content:
        sections = content.split("\n## ")
        for i,section in enumerate(sections):
            section_text = section.strip()
            if not section_text:
                continue
            if i>0:
                section_text = "## "+section_text
            chunk_id = f"{base_name}_chunk_{i+1}"
            chunks.append(DocumentChunk(chunk_id,section_text,file))
    else:
        chunk_id = f"{base_name}::chunk0"
        chunks.append(DocumentChunk(chunk_id,content.strip(),file))
    return chunks

def build_chunk_index():
    all_chunks = []
    docs = load_documents()
    for doc in docs:
        chunks = chunk_doc(doc["filename"],doc["content"])
        all_chunks.extend(chunks)
        
    return all_chunks

class TFIDFRetriever:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_ready = False
    
    def index_docs(self):
        self.chunks = build_chunk_index()
        if not self.chunks:
            print("No documnet found to index")
            return
        
        texts = [chunk.content for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range = (1,2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_ready = True
        print(f"Indexed {len(self.chunks)} chunks from {len(set(c.source_file for c in self.chunks))} documents")
        
    def retrieve(self,query,top_k=3):
        if not self.is_ready:
            print("Index not built. Call index_docs() first.")
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector,self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            chunk = self.chunks[i]
            chunk.score = float(similarities[i])
            results.append(chunk)
        return results
    
retriever = TFIDFRetriever()
def get_retriever():
    if not retriever.is_ready:
        retriever.index_docs()
    return retriever

ret = get_retriever()
print("\nAll indexed chunks:")
for chunk in ret.chunks:
    print(f"  - {chunk.chunk_id} ({len(chunk.content)} chars)")

# Test queries
print("\n" + "=" * 60)
test_queries = [
    "What is the return policy for beverages?",
    "When was Summer Beverages 1997 campaign?",
    "How do you calculate AOV?",
    "What categories are in the catalog?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 40)
    results = ret.retrieve(query, top_k=2)
    for chunk in results:
        print(f"  [{chunk.score:.3f}] {chunk.chunk_id}")
        preview = chunk.content[:80].replace("\n", " ")
        print(f"           {preview}...")

    
