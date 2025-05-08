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
        'pow': pow  # Python's built-in pow function, equivalent to **
    }

    OPERATOR_ARITY = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        '**': 2,  # Changed from '^' to '**'
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
        '**': sympy.Pow, # Changed from '^' to '**', sympy.Pow handles both
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
        Supports standard math operators and functions. Uses '**' for power.
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
                # Use the operator token directly in the infix string
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
            # Use OPERATOR_FUNCS mapping which now includes '**'
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
                # Ensure input prefix uses '**' if converting from external source
                tokens = expression.replace('^', '**').split()
                self.sympy_expression = self.parse_prefix(tokens)
            else:
                # sympy.sympify handles both '^' and '**' and converts them to sympy.Pow
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
                    # Handle symbols that look like x_ but aren't x_number
                     pass # Or raise ValueError(f"Invalid variable name: {symbol.name}") if strict

        computable_expression = str(self.sympy_expression)
        # SymPy's str() method uses '**' for power, so no explicit '^' replacement is needed here.

        for i in range(1, self.max_var + 1):
            # Use regex to match whole words to avoid issues with x_1 followed by x_11
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
        Outputs prefix tokens using '**' for power.
        """
        def traverse(expr):
            if expr.is_Atom:
                return [str(expr)]
            # Handle specific known SymPy functions
            elif expr.func == sympy.Add:
                # Corrected: concatenate the full lists from recursive calls
                tokens = ['+']
                for arg in expr.args:
                    tokens.extend(traverse(arg))
                return tokens
            elif expr.func == sympy.Mul:
                # Corrected: concatenate the full lists from recursive calls
                tokens = ['*']
                for arg in expr.args:
                    tokens.extend(traverse(arg))
                return tokens
            elif expr.func == sympy.Pow:
                # Corrected: concatenate the full lists from recursive calls
                # Need to handle the base and exponent
                if len(expr.args) == 2:
                    base_tokens = traverse(expr.args[0])
                    exp_tokens = traverse(expr.args[1])
                    return ['**'] + base_tokens + exp_tokens
                else:
                    # Handle unexpected number of args for Pow, maybe raise error or fallback
                    print(f"Warning: Unexpected number of arguments for Pow: {expr.args}")
                    op = Expression._get_operator_symbol(expr.func)
                    tokens = [op]
                    for arg in expr.args:
                        tokens.extend(traverse(arg))
                    return tokens

            elif expr.func in (sympy.sin, sympy.cos, sympy.tan, sympy.log, sympy.exp, sympy.sqrt, sympy.Abs):
                op = Expression._get_operator_symbol(expr.func)
                # Corrected: concatenate the full list for the single argument
                return [op] + traverse(expr.args[0])
            else:
                # Handle other potential sympy functions or custom ones (like subtraction/division via Mul/Add)
                op = Expression._get_operator_symbol(expr.func)
                tokens = [op]
                for arg in expr.args:
                    # Corrected: concatenate the full lists for arguments
                    tokens.extend(traverse(arg))
                return tokens


        expr_tree = sympy.sympify(expression)
        prefix_tokens = traverse(expr_tree)
        return ' '.join(prefix_tokens)

    @staticmethod
    def _get_operator_symbol(func):
        # Reverse lookup from OPERATOR_FUNCS, prioritizing lambda functions if needed
        for op, f in Expression.OPERATOR_FUNCS.items():
            if isinstance(f, sympy.FunctionClass) and f == func:
                 return op
            # For lambda functions, we might need to check their __name__ or rely on the sympy type
            if func == sympy.core.power.Pow and op == '**': return '**'
            if func == sympy.core.mul.Mul and op == '*': return '*'
            if func == sympy.core.add.Add and op == '+': return '+'
            # Add checks for other common SymPy function types
            if func == sympy.sin and op == 'sin': return 'sin'
            if func == sympy.cos and op == 'cos': return 'cos'
            if func == sympy.tan and op == 'tan': return 'tan'
            if func == sympy.log and op == 'log': return 'log'
            if func == sympy.exp and op == 'exp': return 'exp'
            if func == sympy.sqrt and op == 'sqrt': return 'sqrt'
            if func == sympy.Abs and op == 'abs': return 'abs'

        # Fallback for other cases, might return the function name string
        return str(func)


    def __str__(self):
        return f"Expression: {self.original_expression}, Max var index: {self.max_var}, Constant count: {self.constant_count}, Best constants: {self.best_constants}"

    def evaluate(self, x, constants=None):
        if constants is None:
            constants = self.best_constants
        try:
            # The computable_expression string generated from sympy uses '**',
            # and Python's eval understands '**'.
            return eval(self.computable_expression, {"__builtins__": None}, {
                "x": x,
                "constants": constants,
                **self.SAFE_FUNCTIONS
            })
        except Exception as e:
            # Provide more context in the error message
            raise RuntimeError(f"Evaluation failed for expression '{self.computable_expression}' with x={x} and constants={constants}: {e}")

    def fit_constants(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.constant_count == 0:
            # Handle potential errors during evaluation
            try:
                y_pred = np.array([self.evaluate(x) for x in X])
                # Handle cases where all y_pred are the same (r2_score would be NaN or inf)
                if np.all(y_pred == y_pred[0]):
                     return 0.0 # Or handle as appropriate for your use case
                return r2_score(y, y_pred)
            except RuntimeError as e:
                 print(f"Evaluation error during fit_constants (no constants): {e}")
                 return -np.inf # Or handle as an invalid fit

        def loss(constants):
            try:
                y_pred = np.array([self.evaluate(x, constants) for x in X])
                # Handle potential NaN or Inf values in y_pred resulting from evaluation
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return np.inf # Indicate a very high loss for invalid outputs
                return np.mean((y - y_pred) ** 2)
            except RuntimeError as e:
                # If evaluation fails during optimization, treat as high loss
                # print(f"Evaluation error during optimization: {e}") # Optional: for debugging
                return np.inf


        # Define bounds for constants. If the original expression implies a power (using ** or ^),
        # potentially bound the constant used as an exponent. This logic is tied to the string
        # representation and might be fragile. A more robust way would be to analyze the sympy tree.
        # Let's keep the original logic but note its limitation.
        # The original code had bounds based on 'pow' or '^' in self.original_expression.
        # Since we are replacing '^' with '**' internally, let's check for '**' or '^'
        # in the original expression string for this specific constant handling logic.
        bounds = [(2, 5) if ('**' in self.original_expression or '^' in self.original_expression) else (None, None)] * self.constant_count

        # Initial guess for constants
        initial_guess = self.best_constants if self.best_constants and len(self.best_constants) == self.constant_count else [1.0] * self.constant_count

        result = minimize(loss, initial_guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            self.best_constants = result.x.tolist()
            # Calculate R^2 with the optimized constants
            try:
                 y_pred = np.array([self.evaluate(x) for x in X])
                 if np.all(y_pred == y_pred[0]):
                      return 0.0 # Or handle as appropriate
                 return r2_score(y, y_pred)
            except RuntimeError as e:
                 print(f"Evaluation error after optimization: {e}")
                 return -np.inf # Indicate failure
        else:
            # Optimization failed, keep the initial constants or handle as an error
            print(f"Optimization failed: {result.message}")
            # Optionally, recalculate R^2 with initial constants or return a specific error value
            return -np.inf # Indicate optimization failure


    def resolved_expression(self):
        """
        Returns the sympy expression with constants (C) replaced by fitted values.
        """
        # Start from the sympy expression tree for a more robust substitution
        resolved_sympy_expr = self.sympy_expression
        c_symbols = sorted([s for s in resolved_sympy_expr.free_symbols if s.name == 'C'], key=lambda s: str(s)) # Ensure consistent order if multiple Cs

        if len(c_symbols) != self.constant_count:
            # This should ideally not happen if constant_count is calculated correctly
             print("Warning: Mismatch between 'C' symbols found and constant_count.")
             # Attempt to substitute based on count anyway, assuming the order is consistent

        subs_dict = {}
        for i, c_symbol in enumerate(c_symbols):
             if i < len(self.best_constants):
                subs_dict[c_symbol] = self.best_constants[i]
             else:
                 # If more C symbols than constants, substitute remaining with 1 or handle error
                 subs_dict[c_symbol] = 1.0 # Default substitution

        resolved_sympy_expr = resolved_sympy_expr.subs(subs_dict)

        # Store and return the resulting sympy expression
        # We could also store a string representation if needed, but the sympy object is more useful
        self._resolved_sympy_expression = resolved_sympy_expr
        return self._resolved_sympy_expression


'''
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 20) * 10  # 20 variables, range [0, 10]

# Define a few target functions using variables and constants
real_expressions = [
    "1 * x_11+x_12",
    "0.312 * sin(x_13) + 0.211 * log(x_4 + 1)",
    "1.23 * x_5**3 + 0.088 * x_16 + 0.00001", # Using ** in real expression
    "sqrt(abs(x_17)) + 0.112 * cos(x_18)",
    "0.123 * exp(-x_19) + x_10"
]

# Goal expressions using C for constants and ** for power
goal_expressions = [
    "C * x_11+x_12",
    "C * sin(x_13) + C * log(x_4 + 1)",
    "C * x_5**C + C * x_16 + C", # Using ** in goal expression
    "sqrt(abs(x_17)) + C * cos(x_18)",
    "C * exp(-x_19) + x_10"
]

# Evaluate R^2 for each expression
for real, goal in zip(real_expressions, goal_expressions):
    print(f"Real Expression: {real}")
    print(f"Goal Expression: {goal}")

    # Generate target y using known constants
    try:
        expr_real = Expression(real)
        y = np.array([expr_real.evaluate(x) for x in X])
    except Exception as e:
        print(f"Error evaluating real expression '{real}': {e}")
        continue

    try:
        expr_goal = Expression(goal)
        r2 = expr_goal.fit_constants(X, y)

        print(f"Fitted Constants: {expr_goal.best_constants}")
        print(f"Resolved Expression (SymPy): {expr_goal.resolved_expression()}")
        print(f"R^2: {r2:.4f}") # Print the R^2 from the fit_constants method

    except Exception as e:
         print(f"Error processing goal expression '{goal}': {e}")


    print("-" * 50)

'''