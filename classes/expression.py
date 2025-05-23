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
            split_expr = self.computable_expression.split('C')
            new_expr = ""
            for i in range(len(split_expr)-1):
                if split_expr[i] == '' and i == 0:
                    new_expr += f'constants[{i}]' + split_expr[i]
                elif split_expr[i] == '' and i != 0:
                    new_expr += split_expr[i] + f'constants[{i}]'
                else:
                    new_expr += split_expr[i] + f'constants[{i}]' + split_expr[i+1]
                    i += 1            
                i += 1
            
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

    def fit_constants(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.constant_count == 0:
            try:
                y_pred = self.evaluate(X) # Vectorized call
                if not np.all(np.isfinite(y_pred)): # Check for NaNs/Infs
                    return -np.inf
                if np.all(y_pred == y_pred[0]) and len(np.unique(y)) > 1: # Avoid R2 issues with constant prediction for non-constant y
                    return 0.0 # Or handle as per specific requirements
                return r2_score(y, y_pred)
            except Exception as e: # Broader catch for any eval issue
                return -np.inf

        def loss(current_constants):

            try:
                y_pred = self.evaluate(X, current_constants)
                
            except Exception as e:
                print(f"Exception during evaluation: {e}")
                return np.inf
            
            if not np.all(np.isfinite(y_pred)):
                return np.inf 
            
            # MSE calculation
            mse = np.mean((y - y_pred) ** 2)
            
            return mse

        bounds = [(-2., 2.)] * self.constant_count
                
        initial_guess = (
            self.best_constants
            if self.best_constants and len(self.best_constants) == self.constant_count
            else [.0] * self.constant_count # Default to 1.0
        )

        # Ensure initial_guess is a flat numpy array
        initial_guess = np.array(initial_guess, dtype=float).flatten()


        # from scipy.optimize import differential_evolution
        # # Step 1: Use Differential Evolution for global exploration
        # print("\n--- Starting Differential Evolution ---")
        # result_de = differential_evolution(loss, bounds,
        #                                    popsize=70,      # Aumente para 50, 70, ou mais
        #                                    maxiter=10000,   # Aumente para 5000, 10000, ou mais
        #                                    strategy='rand1bin', # Tente 'rand1exp' se rand1bin não funcionar
        #                                    tol=1e-7,        # Tolerância mais apertada
        #                                    mutation=(0.8, 1.2), # Experimente valores mais altos
        #                                    recombination=0.5, # Experimente valores mais baixos
        #                                    seed=42,         # Mantém a reproducibilidade
        #                                    disp=True,       # Exibe o progresso
        #                                    polish=False) 

        # if result_de.success:
        #     print(f"\nDifferential Evolution finished successfully. Best raw constants: {result_de.x}, Best MSE: {result_de.fun}")
        #     # Use the result from DE as initial guess for local optimizer
        #     initial_guess_for_minimize = result_de.x
            
        #     # Step 2: (Optional but recommended) Refine with L-BFGS-B
        #     # L-BFGS-B will be applied to the "raw" (non-rounded) values,
        #     # but the loss function internally rounds for discrete ones.
        #     # It might still struggle if the function is too "stepped" from rounding.
        #     print("\n--- Starting L-BFGS-B refinement ---")
        #     result_min = minimize(loss,
        #                           x0=initial_guess_for_minimize,
        #                           method='L-BFGS-B',
        #                           bounds=bounds,
        #                           options={'maxiter': 500, 'ftol': 1e-9, 'disp': True} # More iterations, tighter tolerance
        #     )

        #     if result_min.success:
        #         print(f"\nL-BFGS-B refinement successful. Final raw constants: {result_min.x}, Final MSE: {result_min.fun}")
        #         self.best_constants = list(result_min.x)
        #     else:
        #         print(f"\nL-BFGS-B refinement failed: {result_min.message}. Using Differential Evolution's result.")
        #         self.best_constants = list(result_de.x)
        # else:
        #     print(f"\nDifferential Evolution did not converge successfully: {result_de.message}. Cannot proceed with optimization.")
        #     return -np.inf # Indicate failure
        
        # try:
        #     y_pred = self.evaluate(X) 
        #     if not np.all(np.isfinite(y_pred)):
        #         print("Final evaluation produced non-finite values for R2 score.")
        #         return -np.inf
        #     if len(np.unique(y)) == 1:
        #         if np.allclose(y_pred, y[0]):
        #             return 1.0
        #         else:
        #             return 0.0
        #     return r2_score(y, y_pred)
        # except Exception as e:
        #     print(f"Error calculating final R2: {e}")
        #     return -np.inf
            
        result = minimize(loss, 
                        x0=initial_guess,
                        method='L-BFGS-B',
                        bounds=bounds,
                        #options={'maxiter': 10, 'maxfun': 10, 'disp': True}
        )

        if result.success:
            self.best_constants = result.x.tolist()
            # print(f"Optimization successful. Final loss: {result.fun}") # Optional
            try:
                y_pred = self.evaluate(X) # Uses self.best_constants (vectorized)
                if not np.all(np.isfinite(y_pred)):
                    return -np.inf
                # Refined R2 calculation for edge cases
                if len(np.unique(y)) == 1: # If y is constant
                    if np.allclose(y_pred, y[0]):
                        return 1.0 # Perfect prediction of a constant
                    else:
                        return 0.0 # Or some other metric for imperfect constant prediction
                #return mean_squared_error(y, y_pred) # Use MSE for optimization
                #return mean_absolute_error(y, y_pred) # Use MAE for robustness
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