import sqlite3,os
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT=os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_PATH=os.path.join(PROJECT_ROOT,"data","northwind.sqlite")
_VIEWS_INIT=False

def _init_views(conn):
    global _VIEWS_INIT
    if _VIEWS_INIT:return
    c=conn.cursor()
    views=[
        'CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details"',
        'CREATE VIEW IF NOT EXISTS OrderDetails AS SELECT * FROM "Order Details"'
    ]
    for v in views:
        try:c.execute(v)
        except:pass
    conn.commit()
    _VIEWS_INIT=True

def get_connection():
    conn=sqlite3.connect(DB_PATH)
    conn.row_factory=sqlite3.Row
    _init_views(conn)
    return conn

def get_full_schema():
    return '''TABLES:
- Orders(OrderID,CustomerID,EmployeeID,OrderDate)
- "Order Details"(OrderID,ProductID,UnitPrice,Quantity,Discount)
- Products(ProductID,ProductName,CategoryID,UnitPrice)
- Categories(CategoryID,CategoryName)
- Customers(CustomerID,CompanyName)

JOINS:
- Orders.OrderID = "Order Details".OrderID
- "Order Details".ProductID = Products.ProductID
- Products.CategoryID = Categories.CategoryID

REVENUE: SUM(od.UnitPrice*od.Quantity*(1-od.Discount))
'''

def execute_query(sql):
    conn=get_connection()
    cur=conn.cursor()
    try:
        cur.execute(sql)
        cols=[d[0] for d in cur.description] if cur.description else []
        rows=[dict(zip(cols,r)) for r in cur.fetchall()]
        conn.close()
        return {"success":True,"columns":cols,"rows":rows,"error":None}
    except Exception as e:
        conn.close()
        return {"success":False,"columns":[],"rows":[],"error":str(e)}
