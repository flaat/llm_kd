"""
Classification script using Decision Tree Classifier.

For each dataset in a list, trains a DecisionTreeClassifier with max_depth=4,
evaluates on the test set, and saves results and the fitted model.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

try:
    from dice_ml.utils import helpers
except ImportError:
    print("Warning: dice_ml not found. Adult Income dataset loading may fail.")
    helpers = None

try:
    import torch
    import torch_geometric
except ImportError:
    print("Warning: torch_geometric not found. Graph datasets (cora, pubmed, citeseer) will not be available.")
    torch = None
    torch_geometric = None


class Dataset:
    def __init__(self, name: str, test_size: float = 0.2, random_seed: int = 42):
        self.name: str = name  # Name of the dataset
        self.test_size: float = test_size  # Proportion of the dataset to include in the test split
        self.random_seed: int = random_seed  # Random seed for reproducibility
        self.name_to_drop: list = []  # List of column names to drop from the dataset
        self.name_numerical: list = []  # List of numerical column names in the dataset
        self.name_target: str = ''  # Name of the target column
        self.df: pd.DataFrame = None  # DataFrame to hold the dataset
        self.X_train: pd.DataFrame = None  # DataFrame for training features
        self.X_test: pd.DataFrame = None  # DataFrame for testing features
        self.y_train: pd.Series = None  # Series for training labels
        self.y_test: pd.Series = None  # Series for testing labels
        self.train_dataset: pd.DataFrame = None  # DataFrame for the training dataset including target
        self.test_dataset: pd.DataFrame = None  # DataFrame for the testing dataset including target
        self.label_encoders: dict = {}  # Dictionary to hold label encoders for each column
        self.graph = None
        
        self.load_dataset()

    def load_dataset(self) -> None:
        """
        Load the dataset based on the dataset name provided during initialization.
        This method sets the appropriate attributes for the dataset, including
        the training and testing splits, feature columns, and target column.
        """
        if self.name == "Titanic":
            # Columns to drop from the dataset
            self.name_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
            # Numerical columns in the dataset
            self.name_numerical = ['Age', 'Fare']
            # Target column name
            self.name_target = 'Survived'
            
            # Try multiple possible paths for the Titanic CSV
            possible_paths = [
                'src/explainer/data/titanic/train.csv', 
            ]
            
            df_loaded = False
            for csv_path in possible_paths:
                if os.path.exists(csv_path):
                    self.df = pd.read_csv(csv_path)
                    df_loaded = True
                    break
            
            if not df_loaded:
                raise FileNotFoundError(
                    f"Could not find Titanic dataset CSV. Tried paths: {possible_paths}"
                )
            
            # Drop unnecessary columns
            self.df = self.df.drop(self.name_to_drop, axis=1)
            # Drop rows with missing values
            self.df = self.df.dropna(axis=0)

            # Encode categorical features
            le_sex = LabelEncoder()
            le_embarked = LabelEncoder()
            self.df['Sex'] = le_sex.fit_transform(self.df['Sex'])
            self.df['Embarked'] = le_embarked.fit_transform(self.df['Embarked'])
            self.label_encoders['Sex'] = le_sex
            self.label_encoders['Embarked'] = le_embarked

            # Separate features and target
            y = self.df.iloc[:, 0]

            # Split the dataset into training and testing sets
            self.train_dataset, self.test_dataset, self.y_train, self.y_test = train_test_split(
                self.df, y, test_size=self.test_size, random_state=self.random_seed, stratify=y)
            
            self.train_dataset.reset_index(drop=True, inplace=True)
            self.test_dataset.reset_index(drop=True, inplace=True)

            # Separate features from the target in training and testing sets
            self.X_train = self.train_dataset.drop(columns=[self.name_target])
            self.X_test = self.test_dataset.drop(columns=[self.name_target])

        
        elif self.name == "Adult Income":
            if helpers is None:
                raise ImportError("dice_ml is required for loading Adult Income dataset. Install it with: pip install dice-ml")
            
            self.name_numerical = ["age", "hours_per_week"]
            self.name_target = "income"
            
            self.df = helpers.load_adult_income_dataset()
            
            # Encode categorical features
            categorical_columns = self.df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le
            
            target = self.df["income"]
            
            self.train_dataset, self.test_dataset, self.y_train, self.y_test = train_test_split(self.df,
                                                                            target,
                                                                            test_size=self.test_size,
                                                                            random_state=self.random_seed,
                                                                            stratify=target)

            self.train_dataset.reset_index(drop=True, inplace=True)
            self.test_dataset.reset_index(drop=True, inplace=True)

            # Separate features from the target in training and testing sets
            self.X_train = self.train_dataset.drop(columns=[self.name_target])
            self.X_test = self.test_dataset.drop(columns=[self.name_target])
        
       
        elif self.name == "Diabetes":
            self.name_numerical = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
            self.name_target = "target"

            # Load the diabetes dataset from sklearn
            diabetes = load_diabetes(as_frame=True, scaled=False)
            self.df = diabetes.frame
            # Change the 'sex' column to 'Male' for 1 and 'Female' for 0
            self.df['sex'] = self.df['sex'].apply(lambda x: 'Male' if x == 1 else 'Female')
            # Binarize the target variable based on the average target value
            average_value = self.df[self.name_target].mean()
            self.df[self.name_target] = (self.df[self.name_target] > average_value).astype(int)

            # Encode categorical features if any (in this case, there are none)
            categorical_columns = self.df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le

            target = self.df[self.name_target]

            self.train_dataset, self.test_dataset, self.y_train, self.y_test = train_test_split(self.df,
                                                                                                target,
                                                                                                test_size=self.test_size,
                                                                                                random_state=self.random_seed,
                                                                                                stratify=target)

            self.train_dataset.reset_index(drop=True, inplace=True)
            self.test_dataset.reset_index(drop=True, inplace=True)

            # Separate features from the target in training and testing sets
            self.X_train = self.train_dataset.drop(columns=[self.name_target])
            self.X_test = self.test_dataset.drop(columns=[self.name_target])
        elif self.name == "California Housing":
            self.name_numerical = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
            self.name_target = "MedHouseVal"

            # Load the California Housing dataset from sklearn
            california_housing = fetch_california_housing(as_frame=True)
            self.df = california_housing.frame

            # Binarize the target variable based on the average house value
            average_value = self.df[self.name_target].mean()
            self.df[self.name_target] = (self.df[self.name_target] > average_value).astype(int)

            # Encode categorical features if any (in this case, there are none)
            categorical_columns = self.df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le

            target = self.df[self.name_target]

            self.train_dataset, self.test_dataset, self.y_train, self.y_test = train_test_split(self.df,
                                                                                                target,
                                                                                                test_size=self.test_size,
                                                                                                random_state=self.random_seed,
                                                                                                stratify=target)

            self.train_dataset.reset_index(drop=True, inplace=True)
            self.test_dataset.reset_index(drop=True, inplace=True)

            # Separate features from the target in training and testing sets
            self.X_train = self.train_dataset.drop(columns=[self.name_target])
            self.X_test = self.test_dataset.drop(columns=[self.name_target])

        else:
            raise ValueError("Unsupported dataset")
    
    def revert_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revert the encoding of the encoded features in the given DataFrame.
        """
        for column, le in self.label_encoders.items():
            df[column] = le.inverse_transform(df[column])
        return df


def evaluate_model(model: DecisionTreeClassifier, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate the model on test set and return metrics.
    
    Args:
        model: Trained DecisionTreeClassifier
        X_test: Test features (pandas DataFrame or numpy array)
        y_test: Test labels (pandas Series or numpy array)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert to numpy arrays if needed (sklearn works with both)
    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values
    else:
        y_test_array = y_test
    
    # Make predictions
    y_pred = model.predict(X_test_array)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test_array, y_pred)),
        'precision': float(precision_score(y_test_array, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test_array, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test_array, y_pred, average='weighted', zero_division=0)),
    }
    
    # Classification report
    report = classification_report(y_test_array, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_test_array, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def save_results(results: Dict[str, Any], output_dir: Path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing results
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results JSON
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")


def save_model(model: DecisionTreeClassifier, model_path: Path):
    """
    Save the trained model using pickle.
    
    Args:
        model: Trained DecisionTreeClassifier
        model_path: Path to save the model
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {model_path}")


def main(dataset_names: List[str], max_depth: int = 4):
    """
    Main function to train and evaluate DecisionTreeClassifier on multiple datasets.
    
    Args:
        dataset_names: List of dataset names to process
        max_depth: Maximum depth for the decision tree
    """
    for dataset_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Load dataset
            print(f"Loading dataset: {dataset_name}...")
            dataset = Dataset(name=dataset_name)
            print(f"Loaded {dataset_name}:")
            print(f"  Training samples: {len(dataset.X_train)}")
            print(f"  Test samples: {len(dataset.X_test)}")
            print(f"  Features: {dataset.X_train.shape[1]}")
            
            # Skip graph datasets (they require different handling)
            if dataset.graph is not None:
                print(f"Skipping {dataset_name}: Graph datasets not supported for DecisionTreeClassifier")
                continue
            
            # Train the Decision Tree model
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            model.fit(dataset.X_train, dataset.y_train)
            
            # Evaluate on test set
            results = evaluate_model(model, dataset.X_test, dataset.y_test)
            
            # Add dataset info to results
            results['dataset_name'] = dataset_name
            results['max_depth'] = max_depth
            results['n_train_samples'] = len(dataset.X_train)
            results['n_test_samples'] = len(dataset.X_test)
            results['n_features'] = dataset.X_train.shape[1]
            
            # Save results
            results_dir = Path("src/explainer/clf_results") / dataset_name
            save_results(results, results_dir)
            
            # Save model (replace spaces with underscores for filename)
            safe_name = dataset_name.replace(" ", "_")
            model_path = Path("src/explainer/clf_models") / f"{safe_name}.pkl"
            save_model(model, model_path)
            
            # Print summary
            print(f"\nSummary for {dataset_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # Example usage: list of dataset names
    dataset_names = [
        "Titanic",
        "Adult Income",
        "Diabetes",
        "California Housing"
    ]
    
    max_depth = 4
    
    main(dataset_names, max_depth=max_depth)

