import sqlite3
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_PATH = os.path.join(PROJECT_ROOT,"data","northwind.sqlite")


def get_connection():
    "Creating a connection to the SQLite database"
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row #-> access columns by name
    return conn

def get_all_tables():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def get_table_schema(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    cols = cursor.fetchall()
    
    conn.close()
    
    result = []
    for i in cols:
        result.append({
            "name": i[1],
            "type": i[2],
            "nullable": not i[3],
            "primary_key": i[5] == 1 #-> 1 if primary key,0 if not
        })
    return result

def get_full_schema():
    tables = get_all_tables()
    schema_text = "Database Schema:\n"
    for table in tables:
        cols = get_table_schema(table)
        schema_text += f"Table: {table}\n"
        for i in cols:
            pk_marker = " [PK]" if i["primary_key"] else ""
            schema_text += f"  - {i['name']} ({i['type']}){pk_marker}\n"
        schema_text += "\n"
    
    return schema_text

def execute_query(sql):
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append(dict(zip(columns,row)))
        conn.close()
        
        return{
            "success": True,
            "columns": columns,
            "rows": results,
            "error": None
        }
    except Exception as e:
        conn.close()
        return{
            "success": False,
            "columns": [],
            "rows": [],
            "error": str(e)
        }

def get_sample_data(table_name,limit=7):
    sql = f'SELECT * FROM "{table_name}" LIMIT {limit}'
    return execute_query(sql)
        

print("1.All Tables")
tables = get_all_tables()
for i in tables:
    print(f"- {i}")
    

print("\n2. Table Schema")
schema = get_table_schema("Orders")
for i in schema:
    print(f" - {i}")

print("\n3. Sample Query")
result = execute_query("SELECT ProductID,ProductName,UnitPrice FROM Products LIMIT 3")
if result["success"]:
    for row in result["rows"]:
        print(f" - {row}")
else:
    print(f"Error: {result['error']}")

print("\n4. Full Schema")
full_schema = get_full_schema()
print(full_schema)