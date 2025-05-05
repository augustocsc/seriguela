import sympy
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import math
import re

class Expression:
    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'log': math.log,
        'exp': math.exp,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'abs': abs,
        'pow': pow
    }

    OPERATOR_ARITY = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        '^': 2,
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'log': 1,
        'sqrt': 1,
        'exp': 1
    }

    OPERATOR_FUNCS = {
        '+': sympy.Add,
        '-': lambda x, y: x - y,
        '*': sympy.Mul,
        '/': lambda x, y: x / y,
        '^': sympy.Pow,
        'sin': sympy.sin,
        'cos': sympy.cos,
        'tan': sympy.tan,
        'log': sympy.log,
        'sqrt': sympy.sqrt,
        'exp': sympy.exp
    }

    @staticmethod
    def prefix_to_infix(tokens):
        """
        Convert a prefix expression (list of tokens) to an infix expression string.
        Supports standard math operators and functions.
        """
        if not tokens:
            raise ValueError("Unexpected end of input")

        token = tokens.pop(0)

        if token in Expression.OPERATOR_ARITY:
            arity = Expression.OPERATOR_ARITY[token]
            args = [Expression.prefix_to_infix(tokens) for _ in range(arity)]

            if arity == 1:
                return f"{token}({args[0]})"
            elif arity == 2:
                return f"({args[0]} {token} {args[1]})"
        else:
            return token

    @staticmethod
    def parse_prefix(tokens):
        if not tokens:
            raise ValueError("Unexpected end of input")

        token = tokens.pop(0)

        if token in Expression.OPERATOR_ARITY:
            arity = Expression.OPERATOR_ARITY[token]
            args = [Expression.parse_prefix(tokens) for _ in range(arity)]
            return Expression.OPERATOR_FUNCS[token](*args)
        else:
            try:
                return sympy.sympify(token)
            except:
                return sympy.Symbol(token)

    def __init__(self, expression, is_prefix=False):
        try:
            self.original_expression = expression  # Save original

            if is_prefix:
                tokens = expression.split()
                self.sympy_expression = self.parse_prefix(tokens)
            else:
                self.sympy_expression = sympy.sympify(expression)
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {e}")
        
        self.max_var = 0
        for symbol in self.sympy_expression.free_symbols:
            if symbol.name.startswith('x_'):
                try:
                    index = int(symbol.name.split('_')[1])
                    self.max_var = max(self.max_var, index)
                except ValueError:
                    raise ValueError(f"Invalid variable name: {symbol.name}")
        
        computable_expression = str(self.sympy_expression)
        for i in range(1, self.max_var + 1):
            computable_expression = re.sub(rf'\bx_{i}\b', f'x[{i-1}]', computable_expression)
        self.constant_count = computable_expression.count('C')

        new_expression = ""
        c_index = 0
        i = 0
        while i < len(computable_expression):
            if computable_expression[i] == 'C':
                new_expression += f'constants[{c_index}]'
                c_index += 1
                i += 1
            else:
                new_expression += computable_expression[i]
                i += 1

        self.computable_expression = new_expression
        self.best_constants = [1.0] * self.constant_count

    @staticmethod
    def infix_to_prefix(expression):
        """
        Convert an infix expression string to a prefix expression string.
        """
        def traverse(expr):
            if expr.is_Atom:
                return [str(expr)]
            elif expr.func == sympy.Add:
                return ['+'] + [token for arg in expr.args for token in traverse(arg)]
            elif expr.func == sympy.Mul:
                return ['*'] + [token for arg in expr.args for token in traverse(arg)]
            elif expr.func == sympy.Pow:
                return ['^'] + [traverse(expr.args[0])[0], traverse(expr.args[1])[0]]
            elif expr.func == sympy.sin:
                return ['sin'] + traverse(expr.args[0])
            elif expr.func == sympy.cos:
                return ['cos'] + traverse(expr.args[0])
            elif expr.func == sympy.tan:
                return ['tan'] + traverse(expr.args[0])
            elif expr.func == sympy.log:
                return ['log'] + traverse(expr.args[0])
            elif expr.func == sympy.exp:
                return ['exp'] + traverse(expr.args[0])
            elif expr.func == sympy.sqrt:
                return ['sqrt'] + traverse(expr.args[0])
            elif expr.func == sympy.Abs:
                return ['abs'] + traverse(expr.args[0])
            else:
                # Default binary operators (sub, div, etc.)
                op_map = {
                    sympy.core.power.Pow: '^',
                    sympy.core.mul.Mul: '*',
                    sympy.core.add.Add: '+',
                }
                op = Expression._get_operator_symbol(expr.func)
                return [op] + [traverse(arg)[0] for arg in expr.args]

        expr_tree = sympy.sympify(expression)
        prefix_tokens = traverse(expr_tree)
        return ' '.join(prefix_tokens)

    @staticmethod
    def _get_operator_symbol(func):
        # Reverse lookup from OPERATOR_FUNCS
        for op, f in Expression.OPERATOR_FUNCS.items():
            if f == func:
                return op
        # Handle common lambda fallback cases (e.g., subtraction, division)
        if func == sympy.core.mul.Mul:
            return '*'
        elif func == sympy.core.add.Add:
            return '+'
        elif func == sympy.core.power.Pow:
            return '^'
        elif func.__name__ == 'sin':
            return 'sin'
        elif func.__name__ == 'cos':
            return 'cos'
        elif func.__name__ == 'tan':
            return 'tan'
        elif func.__name__ == 'log':
            return 'log'
        elif func.__name__ == 'sqrt':
            return 'sqrt'
        elif func.__name__ == 'exp':
            return 'exp'
        return func.__name__

    def __str__(self):
        return f"Expression: {self.original_expression}, Max var index: {self.max_var}, Constant count: {self.constant_count}, Best constants: {self.best_constants}"

    def evaluate(self, x, constants=None):
        if constants is None:
            constants = self.best_constants
        try:
            return eval(self.computable_expression, {"__builtins__": None}, {
                "x": x,
                "constants": constants,
                **self.SAFE_FUNCTIONS
            })
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")

    def fit_constants(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.constant_count == 0:
            y_pred = np.array([self.evaluate(x) for x in X])
            return r2_score(y, y_pred)

        def loss(constants):
            y_pred = np.array([self.evaluate(x, constants) for x in X])
            return np.mean((y - y_pred) ** 2)

        bounds = [(2, 5) if 'pow' in self.original_expression else (None, None)] * self.constant_count
        result = minimize(loss, self.best_constants, method='L-BFGS-B', bounds=bounds)

        if result.success:
            self.best_constants = result.x.tolist()
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")

        y_pred = np.array([self.evaluate(x) for x in X])
        return r2_score(y, y_pred)

    def resolved_expression(self):
        """
        Returns the original expression with constants (C) replaced by fitted values.
        """
        resolved = self.original_expression
        for i, value in enumerate(self.best_constants):
            resolved = resolved.replace('C', f'({value:.6f})', 1)
        
        self.resolved_expression = resolved
        self.resolved_expression = sympy.sympify(resolved)
        return self.resolved_expression


'''
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 20) * 10  # 10 variables, range [0, 10]

# Define a few target functions using variables and constants
real_expressions = [
    "1 * x_11+x_12",
    "0.312 * sin(x_13) + 0.211 * log(x_4 + 1)",
    "1.23 * x_5**3 + 0.088 * x_16 + 0.00001",
    "sqrt(abs(x_17)) + 0.112 * cos(x_18)",
    "0.123 * exp(-x_19) + x_10"
]

goal_expressions = [
    "C * x_11+x_12",
    "C * sin(x_13) + C * log(x_4 + 1)",
    "C * x_5**C + C * x_16 + C",
    "sqrt(abs(x_17)) + C * cos(x_18)",
    "C * exp(-x_19) + x_10"
]

# Evaluate R^2 for each expression
for real, goal in zip(real_expressions, goal_expressions):
    # Generate target y using known constants
    expr = Expression(real)
    y = np.array([expr.evaluate(x) for x in X])
    
    expr = Expression(goal)
    expr.fit_constants(X, y)
    print(expr.resolved_expression())
    print(f"R^2: {expr.fit_constants(X, y):.4f}")

    print("" + "-" * 50)

'''