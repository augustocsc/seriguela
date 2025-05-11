import unittest
import numpy as np
import sympy as sp
import os
import sys

# Add path for Expression class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../classes')))
from expression import Expression
from dataset import RegressionDataset
import pickle

class TestExpression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic dataset
        reg = RegressionDataset('./data/evaluate/srsd-feynman_easy/train', 'feynman-i.12.1.txt', delimiter=' ')
        self.X, self.y = reg.get_numpy()
        

    def test_complex_expression(self):
        # Define the complex expression
        goal_expression = "C*(x_1 * x_2 + C)"


        try:
            expr_goal = Expression(goal_expression)
            
            r2 = expr_goal.fit_constants(self.X, self.y)

            resolved_expr = expr_goal.resolved_expression()
            best_constants = expr_goal.best_constants

            # Assert R^2 is reasonable (close to 1 for a good fit)
            self.assertGreater(r2, 0.9, f"R^2 is too low: {r2}")

            # Print results for debugging
            print(f"Fitted Constants: {best_constants}")
            print(f"Resolved Expression (SymPy): {resolved_expr}")
            print(f"R^2: {r2:.4f}")

        except Exception as e:
            self.fail(f"Error processing goal expression '{goal_expression}': {e}")

if __name__ == "__main__":
    unittest.main()