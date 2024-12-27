import os
import json

path = '/home/rmx/workspace/cav2025_2/cav2025/benchmarks/output/'

example_name = "H10"

print(f"#{example_name}")
code = []

expr_id = 0
mono_id = 0
Q_id = 0

code = []
for root, dirs, files in os.walk(path + example_name):
    # print(f"当前目录: {root}")
    # print(f"子文件夹: {dirs}")
    # print(dirs)
    if dirs:
        continue
    
    code.append("# " + root.split('/')[-1])
    for file in files:
        with open('/'.join([root, file]), 'r') as f:
            data = json.load(f)
        
        code.append("## " + data['name'])
        
        expr = data['rational_expression']
        expr = expr.replace('**', '^')
        expr = expr.replace('/', '//')
        for i in range(1, 30):
            expr = expr.replace(f"x{i}", f"X[{i}]")
        code.append(f"P{expr_id} = {expr}")
        
        mono = data['monomial_list'].replace('**', '^')
        for i in range(1, 30):
            mono = mono.replace(f"x{i}", f"X[{i}]")
        
        code.append(f"M{mono_id} = {mono}")
        
        Q = data['Q'].replace("]", ";\n").replace("[", "")
        Q = Q.replace(',', '')
        Q = Q.replace(";\n;", ';')
        code.append(f"Q{Q_id} = [{Q}]")
        
        code.append(f"solve(n, X, P{expr_id}, Q{mono_id}, M{Q_id}, true)")
        expr_id += 1
        mono_id += 1
        Q_id += 1
    
code.append("\n@show SOS_time Newton_time\n")
        
for c in code:
    print(c)
        
