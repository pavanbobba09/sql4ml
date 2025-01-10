from django.shortcuts import render
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf  # Make sure TensorFlow is installed
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def home(request):
    return render(request, 'myapp/home.html')

@csrf_exempt
def process_form(request):
    result = ""
    show_div = False
    if request.method == "POST":
        show_div = True
        # Retrieve data from the form
        sql_statements = request.POST.get("input1")
        features_table = request.POST.get("input2")
        target_table = request.POST.get("input3")
        iterations = int(request.POST.get("input4"))
        learning_rate = float(request.POST.get("input5"))
        db_host = request.POST.get("input6")
        db_user = request.POST.get("input7")
        db_password = request.POST.get("input8")
        db_name = request.POST.get("input9")
        loss_name = request.POST.get("input10")
        result=sql_statements
        result=end_to_end_translate(
        sql_statements,
        features_table,  
        target_table,      
        iterations,
        learning_rate,
        db_host,
        db_user,
        db_password,
        db_name,
        loss_name)
    else:
        return render(request, 'myapp/home.html')
    return render(request, "myapp/home.html", {"show_div": show_div, "result": result})

sql_to_tf_mappings = {
    "avg": "tf.reduce_mean",
    "pow": "tf.pow",
    "log": "tf.math.log",
    "greatest": "tf.maximum",
    "exp": "tf.exp",
    "1/1+exp": "tf.sigmoid",  
    "sum*":"tf.matmul",
}

def extract_view_name(stmt):
    if isinstance(stmt, str) and "create view" in stmt:
        match = re.search(r'create view (\w+)', stmt)
        if match:
            return match.group(1)
    return ""

def extract_select_expressions(stmt):
    if isinstance(stmt, str) and ("select" in stmt or "create view" in stmt):
        match = re.search(r'select (.+?) from', stmt)
        if match:
            return match.group(1).split(' ')
    return []

def create_tf_expression(stmt):
    operators=['+','-','*','/',')','(',',']
    functions=sql_to_tf_mappings.keys()
    ops=[]
    opds=[]
    exp=''
    for st in stmt:
        if st in functions:
            exp=exp+sql_to_tf_mappings[st]
        elif st in operators:
            exp=exp+st
        elif st=='as':
            break
        else:
            st=st.split('.')
            exp=exp+st[0]
    return exp
          
def end_to_end_translate(sql_statements,features_tablename,target_tablename,iterations,learning_rate,db_host,db_user,db_password,db_name,loss_name):
    sql_statements=sql_statements.split(";")
    print(sql_statements)
    view_names=[]
    expressions=[]
    tf_expressions=[]
    for sql in sql_statements:
        if sql=='':
            pass
        view_names.append(extract_view_name(sql))
        exp=extract_select_expressions(sql)
        expressions.append(create_tf_expression(exp))
        tf_expressions.append(view_names[-1]+" = "+expressions[-1])
    tf_commands="\n\t".join(tf_expressions)
    import_statements=f"""
import tensorflow as tf
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
    """
    data_statements=f"""
connection = pymysql.connect(host = "{db_host}",
                               port = int(3306),
                               user = "{db_user}",
                               password = "{db_password}",
                               db = "{db_name}")
query1 = "SELECT * from {features_tablename}"
housing_data = pd.read_sql(query1, connection)
query2 = "SELECT * from {target_tablename}"
target_data = pd.read_sql(query2, connection)
X_train, X_test, y_train, y_test = train_test_split(housing_data,target_data, test_size=0.2, random_state=42)
X_train = tf.constant(X_train.values, dtype=tf.float32)
y_train = tf.constant(y_train.values, dtype=tf.float32)
y_train = tf.reshape(y_train, [-1, 1])
weights = tf.Variable(tf.random.normal([X_train.shape[1], 1]), name="weights")
bias = tf.Variable(tf.random.normal([1]), name="bias")
learning_rate = {learning_rate}
optimizer = tf.optimizers.Adam(learning_rate)
    
for epoch in range({iterations}):
    with tf.GradientTape() as tape:
        {tf_commands}
    gradients = tape.gradient({loss_name}, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))
       
    if epoch % 100 == 0:
       print(f"Epoch: ",epoch, "Loss:",{loss_name}.numpy())
    """
    return import_statements+data_statements