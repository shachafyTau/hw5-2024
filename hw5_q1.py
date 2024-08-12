import pathlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Tuple

class QuestionnaireAnalysis:
    def __init__(self, data_fname: Union[pathlib.Path, str]):
        try:
            self.data_fname = pathlib.Path(data_fname).resolve()
        except TypeError:
            raise ValueError("Please supply a valid type.")
        
        if not self.data_fname.exists():
            raise ValueError(f"File {self.data_fname} does not exist.")

    def read_data(self):
        try:
            self.data = pd.read_json(self.data_fname)
        except ValueError:
            raise ValueError(f"File {self.data_fname} is not a valid JSON file.")
        except FileNotFoundError:
            raise ValueError(f"File {self.data_fname} not found.")

    def show_age_distrib(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.data is None:
            raise ValueError("No data. Call read_data().")
        
        bins = np.linspace(0, 100, 11)
        _, ax = plt.subplots()
        hist, bin_edges, _ = ax.hist(self.data["age"], bins=bins, edgecolor='black', alpha=0.7)

        ax.set_xlabel("Age")
        ax.set_ylabel("Counts")
        ax.set_title("Age distribution of the participants")
        plt.xticks(bins)
        plt.grid(True)
        plt.show()

        return hist, bin_edges

    def remove_rows_without_mail(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data. Call read_data().")
        
        data = pd.DataFrame(self.data)
        
        def is_valid_email(email: str) -> bool:
            if email.count('@') != 1 or email.count('.') != 1 or email.startswith('@') or email.startswith('.') or email.endswith('@'):
                return False
            _, domain = email.split('@')
            if not domain or domain.startswith('.') or domain.endswith('.'):
                return False
            return True
        
        data = data[data['email'].apply(is_valid_email)]
        data.reset_index(drop=True, inplace=True)
        
        return data
    
    def fill_na_with_mean(self) -> Tuple[pd.DataFrame, np.ndarray]:
        if self.data is None:
            raise ValueError("No data. Call read_data().")
        
        data = pd.DataFrame(self.data)
        question_pattern = re.compile(r'^q\d+$')  
        
        grade_columns = [col for col in data.columns if question_pattern.match(col)]
        data[grade_columns] = data[grade_columns].apply(pd.to_numeric, errors='coerce')
        
        nan_rows = data[grade_columns].isna().any(axis=1)
        indices_corrected = data[nan_rows].index.to_numpy()
        
        def fill_na(row):
            row_mean = row.mean()
            return row.fillna(row_mean)
        
        data[grade_columns] = data[grade_columns].apply(fill_na, axis=1)
        return data, indices_corrected

    def score_subjects(self, maximal_nans_per_sub: int = 1) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data. Call read_data() first.")
        data = pd.DataFrame(self.data)
        
        question_pattern = re.compile(r'^q\d+$')
        grade_columns = [col for col in data.columns if question_pattern.match(col)]
        
        data[grade_columns] = data[grade_columns].apply(pd.to_numeric, errors='coerce')
        
        def calculate_score(row):
            num_nans = row[grade_columns].isna().sum()
            if num_nans > maximal_nans_per_sub:
                return np.nan
            scores = row[grade_columns].dropna()
            if not scores.empty:
                return np.floor(scores.mean())
            return np.nan
        
        data['score'] = data.apply(calculate_score, axis=1).astype('UInt8')
        return data
    
    def correlate_gender_age(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data. Call read_data().")
        
        data = pd.DataFrame(self.data)
        
        question_pattern = re.compile(r'^q\d+$')
        grade_columns = [col for col in data.columns if question_pattern.match(col)]
        
        data[grade_columns] = data[grade_columns].apply(pd.to_numeric, errors='coerce')
        data['age'] = pd.to_numeric(data['age'], errors='coerce')

        data = data.dropna(axis=0, subset=['age'])
        data['age'] = data['age'].apply(lambda x: x > 40 )
        
        data.set_index(['gender', 'age'], inplace=True)
        grouped = data[grade_columns].groupby(['gender', 'age']).mean()
        
        return grouped


if __name__ == "__main__":
    t = QuestionnaireAnalysis(data_fname="data.json")
    t.read_data()  
    result_df = t.correlate_gender_age()
    print(result_df)
