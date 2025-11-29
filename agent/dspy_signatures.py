import dspy
try:from dspy.teleprompt import BootstrapFewShot
except:
    try:from dspy.teleprompter import BootstrapFewShot
    except:BootstrapFewShot=None

def setup_dspy():
    lm=dspy.LM(model="ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M",api_base="http://localhost:11434",api_key="",temperature=0.0)
    dspy.configure(lm=lm)
    return lm

class RouterSignature(dspy.Signature):
    """Classify question: rag (doc lookup), sql (database query), hybrid (both)"""
    question=dspy.InputField()
    route=dspy.OutputField(desc="One of: rag, sql, hybrid")

class PlannerSignature(dspy.Signature):
    """Extract constraints from question and docs"""
    question=dspy.InputField()
    doc_context=dspy.InputField()
    date_range=dspy.OutputField(desc="YYYY-MM-DD to YYYY-MM-DD or none")
    filters=dspy.OutputField(desc="Category/product filters or none")
    kpi_formula=dspy.OutputField(desc="Formula from docs or none")

class SynthesizerSignature(dspy.Signature):
    """Generate final answer matching format_hint exactly"""
    question=dspy.InputField()
    format_hint=dspy.InputField()
    doc_context=dspy.InputField()
    sql_result=dspy.InputField()
    final_answer=dspy.OutputField()
    explanation=dspy.OutputField()
    citations=dspy.OutputField()

class Router(dspy.Module):
    RAG_KW=["policy","return window","according to"]
    SQL_KW=["top","revenue","total","average","sum","margin","aov"]
    def __init__(self):
        super().__init__()
        self.predict=dspy.Predict(RouterSignature)
    def forward(self,question):
        q=question.lower()
        for kw in self.RAG_KW:
            if kw in q:return "rag"
        has_sql=any(kw in q for kw in self.SQL_KW)
        has_doc=any(x in q for x in ["1997","campaign","calendar","kpi","definition"])
        if has_sql and has_doc:return "hybrid"
        if has_sql:return "sql"
        try:
            r=self.predict(question=question).route.lower().strip()
            if "rag" in r:return "rag"
            if "sql" in r:return "sql"
            return "hybrid"
        except:return "hybrid"

class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict=dspy.Predict(PlannerSignature)
    def forward(self,question,doc_context):
        try:
            r=self.predict(question=question,doc_context=doc_context)
            return {"date_range":getattr(r,"date_range","none"),"filters":getattr(r,"filters","none"),"kpi_formula":getattr(r,"kpi_formula","none")}
        except:return {"date_range":"none","filters":"none","kpi_formula":"none"}

class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict=dspy.ChainOfThought(SynthesizerSignature)
    def forward(self,question,format_hint,doc_context,sql_result):
        try:
            r=self.predict(question=question,format_hint=format_hint,doc_context=doc_context,sql_result=sql_result)
            return {"answer":getattr(r,"final_answer",""),"explanation":getattr(r,"explanation",""),"citations":getattr(r,"citations","")}
        except:return {"answer":"","explanation":"Error","citations":""}

# DSPy Optimization for Router
ROUTER_TRAIN=[
    dspy.Example(question="What is the return policy for beverages?",route="rag").with_inputs("question"),
    dspy.Example(question="Top 3 products by revenue all time",route="sql").with_inputs("question"),
    dspy.Example(question="Revenue during Summer Beverages 1997 campaign",route="hybrid").with_inputs("question"),
    dspy.Example(question="According to product policy, return window for unopened beverages?",route="rag").with_inputs("question"),
    dspy.Example(question="Average order value during Winter Classics 1997",route="hybrid").with_inputs("question"),
    dspy.Example(question="Best customer by gross margin in 1997",route="hybrid").with_inputs("question"),
    dspy.Example(question="Total quantity sold in Beverages category June 1997",route="hybrid").with_inputs("question"),
    dspy.Example(question="What categories are in the catalog?",route="rag").with_inputs("question"),
]

def router_metric(example,pred,trace=None):
    return float(getattr(pred,"route","").lower().strip()==example.route)

_OPT_ROUTER=None
def get_optimized_router():
    global _OPT_ROUTER
    if _OPT_ROUTER:return _OPT_ROUTER
    base=dspy.Predict(RouterSignature)
    if not BootstrapFewShot or not ROUTER_TRAIN:
        _OPT_ROUTER=base
        return _OPT_ROUTER
    try:
        opt=BootstrapFewShot(metric=router_metric,max_bootstrapped_demos=3,max_labeled_demos=3,max_rounds=1)
        _OPT_ROUTER=opt.compile(base,trainset=ROUTER_TRAIN)
        print("[DSPy] Router optimized")
    except Exception as e:
        print(f"[DSPy] Optimization failed: {e}")
        _OPT_ROUTER=base
    return _OPT_ROUTER
