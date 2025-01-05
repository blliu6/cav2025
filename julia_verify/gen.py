import os
import json
import argparse

parser = argparse.ArgumentParser(description="Process example name.")

# 添加命令行参数
parser.add_argument(
    "-e",
    "--example_name",
    type=str,
    default="C18",  # 设置默认值
    help="Name of the example to process (default: C18)"
)

parser.add_argument(
    "-i",
    "--index",
    type=str,
    default="1",  # 设置默认值
    help="null"
)
args = parser.parse_args()

path = '../benchmarks/output/'

example_name = args.example_name

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

file = open(f"./{example_name}_{args.index}.jl", 'w')

for c in code:
    print(c, file=file)
