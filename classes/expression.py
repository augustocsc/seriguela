import sympy
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import math
import re

    
class Expression:
    SAFE_FUNCTIONS = {
        'sqrt': np.sqrt,
        'log': np.log,
        'exp': np.exp,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'asin': np.arcsin, # Corrected to np.arcsin
        'abs': np.abs,
        'pow': np.power, # Use np.power for vectorization and NaN handling
        # '**' is handled by Python's eval; if operands are numpy arrays, np.power is used.
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

    def parse_prefix(self, tokens):
        """Parse prefix notation expression to SymPy.

        Example: ['*', 'x_1', '+', 'x_2', 'C'] -> x_1*(x_2 + C)
        """
        if not tokens:
            raise ValueError("Empty token list")

        # Define unary and binary operators
        UNARY_OPS = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'asin'}
        BINARY_OPS = {'+', '-', '*', '/', '**', '^'}

        stack = []

        # Process tokens in reverse order
        for token in reversed(tokens):
            if token in BINARY_OPS or token in UNARY_OPS:
                # Operator: pop operands from stack
                if token in UNARY_OPS:
                    if len(stack) < 1:
                        raise ValueError(f"Not enough operands for {token}")
                    arg = stack.pop()
                    if token in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'asin']:
                        stack.append(f"{token}({arg})")
                    else:
                        raise ValueError(f"Unknown unary operator: {token}")
                else:  # Binary operator
                    if len(stack) < 2:
                        raise ValueError(f"Not enough operands for {token}")
                    right = stack.pop()
                    left = stack.pop()

                    # Handle operator mapping
                    op_map = {'+': '+', '-': '-', '*': '*', '/': '/', '**': '**', '^': '**'}
                    op = op_map.get(token, token)

                    if op in ['**', '^']:
                        stack.append(f"({left})**({right})")
                    elif op == '/':
                        stack.append(f"({left})/({right})")
                    else:
                        stack.append(f"({left}){op}({right})")
            else:
                # Operand: push to stack
                stack.append(token)

        if len(stack) != 1:
            raise ValueError(f"Invalid prefix expression, {len(stack)} elements remaining")

        return sympy.sympify(stack[0], evaluate=False)

    def __init__(self, expression, is_prefix=False):
        try:
            self.original_expression = expression  # Save original

            if is_prefix:
                # Ensure input prefix uses '**' if converting from external source
                tokens = expression.replace('^', '**').split()
                self.sympy_expression = self.parse_prefix(tokens)
            else:
                # Load the expression as a sympy expression without simplification
                self.sympy_expression = sympy.sympify(expression, evaluate=False)
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

        for i in range(1, self.max_var + 1):
            # Use regex to match whole words to avoid issues with x_1 followed by x_11
            computable_expression = re.sub(rf'\bx_{i}\b', f'x[{i-1}]', computable_expression)
        

        self.computable_expression = computable_expression.replace('**C', '**2')
        
        self.constant_count = self.computable_expression.count('C')
        self.best_constants = [1.0] * self.constant_count


        if self.constant_count > 0:
            # Replace 'C' with indexable constants
            split_expr = self.computable_expression.split('C')
            new_expr = split_expr[0]  # Start with first part

            for i in range(1, len(split_expr)):
                # Add constant reference
                new_expr += f'constants[{i-1}]'
                # Add next part
                new_expr += split_expr[i]

            self.computable_expression = new_expr
            


        

    def __str__(self):
        return f"Expression: {self.original_expression}, Best constants: {self.best_constants}"
    def sympy_str(self):
        """
        Returns the string representation of the sympy expression.
        """
        return str(self.sympy_expression)
    
    def is_valid_on_dataset(self, X, test_constants_list=None):
        """
        Checks if the expression evaluates to valid (finite) values for all rows in X,
        across one or more sets of test constants.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features)
            test_constants_list (list of lists): Optional. Defaults to [[1.0]*count].
                Example: [[1.0]*n, [0.5]*n, [2.0]*n] to test more thoroughly.

        Returns:
            bool: True if no evaluation returns nan/inf or crashes. False otherwise.
        """
        if test_constants_list is None:
            test_constants_list = [[1.0] * self.constant_count]
        
        try:
            for constants in test_constants_list:
                results = self.evaluate(X, constants)
                
                if not np.all(np.isfinite(results)):
                    return False
            
            return True
        except Exception:
            return False

    # Inside the Expression class
    def evaluate(self, X, constants=None):
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)  # Hide power/tan warnings
        #     np.seterr(invalid='ignore', divide='ignore') 

            

        if constants is None:
            # print("No constants provided, using best constants.") # Optional: uncomment for debugging
            constants = self.best_constants

        try:
            local_env = {
                "constants": np.array(constants), # Ensure constants is a numpy array for broadcasting
                **self.SAFE_FUNCTIONS,
                "__builtins__": None
            }

            if not isinstance(X, np.ndarray):
                X = np.array(X) # Ensure X is a numpy array
            
            # Ensure X is 2D, even if it has only one sample
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # x becomes a list of columns (1D arrays of shape (n_samples,))
            x_cols = [X[:, i] for i in range(X.shape[1])]
            local_env["x"] = x_cols
            
            # The result will be a numpy array of shape (n_samples,)
            
            try:
                y_pred_array = eval(self.computable_expression, local_env)

            except FloatingPointError as e:
                # print(f"FloatingPointError during eval: {e}")
                # print(f"Expression: {self.computable_expression}")
                # print(f"Constants: {constants}")
                return np.full(X.shape[0], np.nan)  # Return NaNs to be caught by loss

            except Exception as e:
                # print(f"General exception during eval: {e}")
                return np.full(X.shape[0], np.nan)

            finally:
                np.seterr(all='warn')  # 🔁 Reset to default behavior

            # Ensure output is float to avoid issues with mixed types if some results are int
            return np.asarray(y_pred_array, dtype=float)

        except Exception as e:
            # Return an array of NaNs of the expected shape to ensure loss calculation doesn't break
            num_samples = X.shape[0] if X.ndim > 0 else 1
            return np.full(num_samples, np.nan) # Return NaNs on error

    def fit_constants(self, X, y, optimize=False):
        """Evaluate expression with constants and return R² score.

        Args:
            X: Input data array
            y: Target values
            optimize: If True, optimize constants using L-BFGS-B.
                     If False (default), use C=1 for all constants.

        Returns:
            R² score (or -inf if evaluation fails)
        """
        X = np.array(X)
        y = np.array(y)

        # Use C=1 for all constants (no optimization by default)
        self.best_constants = [1.0] * self.constant_count

        if not optimize or self.constant_count == 0:
            # No optimization - just evaluate with C=1
            try:
                y_pred = self.evaluate(X)
                if not np.all(np.isfinite(y_pred)):
                    return -np.inf
                if np.all(y_pred == y_pred[0]) and len(np.unique(y)) > 1:
                    return 0.0
                return r2_score(y, y_pred)
            except Exception as e:
                return -np.inf

        # Only reaches here if optimize=True and constant_count > 0
        def loss(current_constants):
            try:
                y_pred = self.evaluate(X, current_constants)
            except Exception as e:
                return np.inf

            if not np.all(np.isfinite(y_pred)):
                return np.inf

            return np.mean((y - y_pred) ** 2)

        bounds = [(-2., 2.)] * self.constant_count
        initial_guess = np.array([1.0] * self.constant_count, dtype=float)

        result = minimize(loss,
                        x0=initial_guess,
                        method='L-BFGS-B',
                        bounds=bounds)

        if result.success:
            self.best_constants = result.x.tolist()
            try:
                y_pred = self.evaluate(X)
                if not np.all(np.isfinite(y_pred)):
                    return -np.inf
                if len(np.unique(y)) == 1:
                    if np.allclose(y_pred, y[0]):
                        return 1.0
                    else:
                        return 0.0
                return r2_score(y, y_pred)
            except Exception as e:
                return -np.inf
        else:
            return -np.inf

# from dataset import RegressionDataset

# import numpy as np
# import warnings

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=RuntimeWarning)
#     np.seterr(invalid='ignore')

# #reg = RegressionDataset('../data/evaluate/srsd-feynman_hard/train', 'feynman-bonus.12.txt', delimiter=' ')
# reg = RegressionDataset('./data/evaluate/srsd-feynman_easy/train', 'feynman-i.18.16.txt', delimiter=' ')
# X, y = reg.get_numpy()

# #x = np.array(X).T
# expression = "x_1*x_2*sin(x_4)"
# #expr = "0.5*x[0]*x[1]**2"


# expr = Expression(expression)
# print("Expression:", expr)

# if expr.is_valid_on_dataset(X):
#     print("Expression is valid on dataset.")
#     score = expr.fit_constants(X, y)
#     print("Fitted constants:", expr.best_constants)
#     print("R2 score:", score)
# else:
#     print("Expression is not valid on dataset.")