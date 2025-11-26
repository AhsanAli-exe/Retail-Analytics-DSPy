import dspy

def setup_dspy():
    lm = dspy.LM(
        model="ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M",
        api_base="http://localhost:11434",
        api_key=""
    )
    dspy.configure(lm=lm)
    return lm

class RouterSignature(dspy.Signature):
    question = dspy.InputField(desc="The user's question about retail analytics")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")
    
    
class TexttoSQLSignature(dspy.Signature):
    question = dspy.InputField(desc="Natural language question")
    schema = dspy.InputField(desc="Database schema information")
    constraints = dspy.InputField(desc="Any constraints like date ranges or filters")
    sql_query = dspy.OutputField(desc="Valid SQLite query")
    
class SynthesizerSignature(dspy.Signature):
    question = dspy.InputField(desc="Original user question")
    format_hint = dspy.InputField(desc="Required output format like 'int' or '{category:str, quantity:int}'")
    doc_context = dspy.InputField(desc="Retrieved document chunks")
    sql_result = dspy.InputField(desc="SQL query results (if any)")
    final_answer = dspy.OutputField(desc="Answer matching the format_hint exactly")
    explanation = dspy.OutputField(desc="Brief 1-2 sentence explanation")
    citations = dspy.OutputField(desc="List of sources used: table names and chunk IDs")
    
class PlannerSignature(dspy.Signature):
    question = dspy.InputField(desc="User question")
    doc_context = dspy.InputField(desc="Retrieved document chunks that may contain constraints")
    date_range = dspy.OutputField(desc="Date range if mentioned, format: 'YYYY-MM-DD to YYYY-MM-DD' or 'none'")
    filters = dspy.OutputField(desc="Any category, product, or customer filters")
    kpi_formula = dspy.OutputField(desc="KPI formula if referenced, or 'none'")
    
