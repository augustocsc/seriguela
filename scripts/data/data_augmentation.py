# augmentor.py

import random
import re

ALL_OPERANDS = ['+', '-', '*', '/', 'log', 'exp', 'cos', 'sqrt', 'asin', 'sin', 'pow', 'tan', 'abs']

def extract_operators(expr_str):
    ops = set()
    if 'exp' in expr_str: ops.add('exp')
    if 'log' in expr_str: ops.add('log')
    if 'cos' in expr_str: ops.add('cos')
    if 'sin' in expr_str: ops.add('sin')
    if 'pow' in expr_str: ops.add('pow')
    if 'sqrt' in expr_str: ops.add('sqrt')
    if 'asin' in expr_str: ops.add('asin')
    if 'tan' in expr_str: ops.add('tan')
    if 'abs' in expr_str: ops.add('abs')
    if '/' in expr_str: ops.add('/')
    for op in ['+', '-', '*']:
        if op in expr_str: ops.add(op)
    return list(ops)

def infer_max_var(expr_str):
    matches = re.findall(r'x_(\d+)', expr_str)
    return max([int(m) for m in matches]) if matches else 1

def generate_expression_instructions(expr_str):
    max_var = infer_max_var(expr_str)
    
    variables = [f"x_{i}" for i in range(1, max_var + random.randint(1, (max_var) + 1))]
    
    used_ops = extract_operators(expr_str)
    extra_ops = list(set(ALL_OPERANDS) - set(used_ops))
    added_ops = random.sample(extra_ops, random.randint(1, len(extra_ops))) if extra_ops else []
    all_ops = sorted(set(used_ops + added_ops))
    constants = ['C']
    wrapped_expr = f"<|startofex|>{expr_str}<|endofex|>"

    return {
        "Simple_Instruct": f"Instruction: Generate a mathematical expression using variables {variables} and operands {all_ops} and {constants} as constant.\nExpression: {wrapped_expr}",
        "Key_Value": f"Variables: {variables}\nOperands: {all_ops}\nConstant: {constants}\nExpression: {wrapped_expr}",
        "Delimiter_Based": f"Input: Variables={variables}, Operands={all_ops}, Constant={constants}\nOutput: {wrapped_expr}",
        "Minimalist": f"{variables} | {all_ops} | {constants} => {wrapped_expr}"
    }


#print(generate_expression_instructions("x_1 - (x_4 - C)*(x_3 + exp(C*x_2) + C)"))