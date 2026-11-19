import pandas as pd
import torch

class RegressionDataset:
    def __init__(self, path: str, file_name: str = 'train.csv', delimiter: str = ',', header: int = 0, 
                 encoding: str = 'utf-8', target_col: str = None):
        """
        Initializes the RegressionDataset by loading data from a CSV file.

        Args:
            path (str): Path to the directory containing the CSV file.
            file_name (str): Name of the CSV file. Defaults to 'train.csv'.
            delimiter (str): Delimiter used in the CSV file. Defaults to ','.
            header (int): Row number to use as the column names. Defaults to 0.
            encoding (str): Encoding of the CSV file. Defaults to 'utf-8'.
            target_col (str): Name of the target column. If None, the last column is used.
        """
        self.data = pd.read_csv(f"{path}/{file_name}", delimiter=delimiter, header=header, encoding=encoding)
        
        if self.data.empty:
            raise ValueError("CSV file is empty.")
        
        if target_col is None:
            target_col = self.data.columns[-1]
        
        if target_col not in self.data.columns:
            raise ValueError(f"CSV must contain a column named '{target_col}'.")
        
        self.X = self.data.drop(columns=[target_col]).apply(pd.to_numeric, errors='coerce').values
        
        self.y = pd.to_numeric(self.data[target_col], errors='coerce').values

    def get_data(self):
        """
        Returns the data as PyTorch tensors (X, y).
        """
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        return X_tensor, y_tensor

    def get_numpy(self):
        """
        Returns the data as NumPy arrays (useful for sympy and R² calculations).
        """
        return self.X, self.y



