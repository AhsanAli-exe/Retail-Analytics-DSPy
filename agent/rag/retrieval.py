import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT=os.path.dirname(os.path.dirname(SCRIPT_DIR))
DOCS_DIR=os.path.join(PROJECT_ROOT,"docs")

class DocumentChunk:
    def __init__(self,chunk_id,content,source):
        self.chunk_id=chunk_id
        self.content=content
        self.source=source
        self.score=0.0

def load_and_chunk():
    chunks=[]
    for f in os.listdir(DOCS_DIR):
        if not f.endswith(".md"):continue
        path=os.path.join(DOCS_DIR,f)
        with open(path,"r",encoding="utf-8") as file:
            content=file.read()
        base=f.replace(".md","")
        if "\n## " in content:
            parts=content.split("\n## ")
            for i,p in enumerate(parts):
                txt=p.strip()
                if not txt:continue
                if i>0:txt="## "+txt
                chunks.append(DocumentChunk(f"{base}::chunk{i}",txt,f))
        else:
            chunks.append(DocumentChunk(f"{base}::chunk0",content.strip(),f))
    return chunks

class TFIDFRetriever:
    def __init__(self):
        self.chunks=[]
        self.vectorizer=None
        self.matrix=None
        self.ready=False
    
    def index(self):
        self.chunks=load_and_chunk()
        if not self.chunks:return
        texts=[c.content for c in self.chunks]
        self.vectorizer=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2))
        self.matrix=self.vectorizer.fit_transform(texts)
        self.ready=True
        print(f"Indexed {len(self.chunks)} chunks")
    
    def retrieve(self,query,top_k=3):
        if not self.ready:return []
        qv=self.vectorizer.transform([query])
        sims=cosine_similarity(qv,self.matrix)[0]
        top_idx=sims.argsort()[::-1][:top_k]
        results=[]
        for i in top_idx:
            c=self.chunks[i]
            c.score=float(sims[i])
            results.append(c)
        return results

_retriever=TFIDFRetriever()
def get_retriever():
    if not _retriever.ready:_retriever.index()
    return _retriever
