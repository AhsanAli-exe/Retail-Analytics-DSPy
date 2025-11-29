import json,click,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from agent.graph_hybrid import run_agent
from agent.dspy_signatures import setup_dspy

def load_questions(path):
    qs=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try:qs.append(json.loads(line))
                except:pass
    return qs

def save_results(results,path):
    with open(path,"w",encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r)+"\n")

@click.command()
@click.option("--batch",required=True,help="Input JSONL file")
@click.option("--out",required=True,help="Output JSONL file")
def main(batch,out):
    print("="*60)
    print("Retail Analytics Copilot")
    print("="*60)
    print("\nInitializing DSPy...")
    setup_dspy()
    print(f"\nLoading questions from: {batch}")
    questions=load_questions(batch)
    print(f"Found {len(questions)} questions\n")
    results=[]
    for i,q in enumerate(questions):
        print(f"{'='*60}\nQuestion {i+1}/{len(questions)}: {q['id']}\n{'='*60}")
        try:
            out_data,trace=run_agent(q["id"],q["question"],q.get("format_hint","str"))
            out_data["id"]=q["id"]
            results.append(out_data)
        except Exception as e:
            print(f"Error: {e}")
            results.append({"id":q["id"],"final_answer":None,"sql":"","confidence":0.0,"explanation":f"Error:{e}","citations":[]})
    print(f"\n{'='*60}\nSaving results to: {out}")
    save_results(results,out)
    print(f"Done! Processed {len(results)} questions.")

if __name__=="__main__":
    main()
