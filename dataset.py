import pandas as pd
import numpy as np
from scipy.io import arff
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset


# Local Code
from config import *


class TabularDataset(Dataset):
    """
    A PyTorch Dataset class for tabular data handling, including preprocessing,
    augmentation, and stratified train-validation-test splitting.

    Attributes:
        - df: The data as pd.DataFrame object
        - meta: Metadata on the df (columns and types)
        - features: Processed feature DataFrame (normalized and one-hot encoded).
        - targets: Target labels as a categorical variable.
        - num_features: Number of features after preprocessing.
        - cat_column_indices: A list of indices for all one-hot encoded columns.
    """
    def __init__(self, file_path, target_column, augment=False, info=False):
        """
        Initialize the dataset object.

        Args:
            file_path (str): Path to the ARFF file containing the dataset.
            target_column (str): Column name of the target variable.
            augment (bool): Whether to perform augmentation on the training set.
            info (bool): Whether to display initial dataset information.
        """
        self.target_column = target_column
        self.augment = augment
        self.cat_column_indices = []
        self.features = []
        self.targets = []

        # Load ARFF file
        data, self.meta = arff.loadarff(file_path)
        self.df = pd.DataFrame(data)

        # Display initial dataset information if requested
        if info:
            self.__print_info()

        # Preprocess the data (normalization and encoding)
        self.__preprocess()

    def __one_hot_encode(self):
        """
        Custom one-hot encoding function for categorical columns.
        - '?' is treated as 'Unknown'.
        - Existing categorical values are mapped to their respective columns.
        - Null values are set to 0.
        - The one-hot encoded columns are normalized to range [-1, 1].
        """
        # Separate continuous and categorical columns
        cont_cols = self.df.select_dtypes(include=["number"]).columns
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns

        # Remove the target column from categorical columns if it exists
        cat_cols = [col for col in cat_cols if col != self.target_column]

        # Initialize a df with numerical columns to build on cat columns
        # Sort columns: numeric first, then categorical (one-hot encoded)
        encoded_df = self.df[cont_cols].copy()

        # One-hot encode categorical columns
        for col in cat_cols:
            # Replace '?' with 'Unknown'
            self.df[col] = self.df[col].replace('?', 'Unknown')
            unique_values = self.df[col].unique()

            # Generate one-hot columns and store the indices
            one_hot_columns = []
            for value in unique_values:
                # Create a new column for the value
                encoded_col = (self.df[col] == value).astype(int)
                one_hot_columns.append(encoded_col)

            # Concatenate the one-hot columns from 1 original cat column
            one_hot_df = pd.concat(one_hot_columns, axis=1)
            one_hot_df = one_hot_df * 2 - 1  # Normalize to [-1, 1]

            # Update the encoded dataframe with the one-hot columns
            encoded_df = pd.concat([encoded_df, one_hot_df], axis=1)

            # Store the indices of the one-hot encoded columns
            self.cat_column_indices.append(list(range(len(encoded_df.columns) - len(one_hot_columns), len(encoded_df.columns))))

        # Save the updated dataframe
        self.features = encoded_df

    def __preprocess(self):
        """
        Preprocess the dataset:
        - Normalize continuous features using MinMax scaling.
        - One-hot encode categorical features manually, treating '?' as a distinct category.
        
        All vector inputs are normalized to [-1,1] range to ensure consistency.
        """
        print("[Dataset Status]: Preprocessing dataset...")

        # Separate target column and features
        self.targets = self.df[self.target_column].astype("category").cat.codes

        all_features = self.df.drop(columns=[self.target_column])

        # Normalize continuous features
        cont_cols = all_features.select_dtypes(include=["number"]).columns
        scaler = MinMaxScaler(feature_range=(-1, 1))  # Modify the range to [-1, 1]
        self.df[cont_cols] = scaler.fit_transform(all_features[cont_cols])

        # One-hot encode categorical features manually (ignoring target column)
        # Will modify self.df and self.features
        self.__one_hot_encode()
        
        self.num_features = self.features.shape[1]

    def __print_info(self):
        """
        Print out information about the dataset, such as column names, data types,
        and unique values for categorical columns.
        """
        print("[Dataset Info]: Dataset loaded and preprocessing completed.")
        print(f"Columns in the dataset: {self.df.columns.tolist()}")
        print(f"Target column: {self.target_column}")
        print(f"Number of features (columns): {self.df.shape[1]}")
        
        # Print unique values per categorical column
        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            unique_values = self.df[col].unique()
            print(f"Column '{col}': {len(unique_values)} unique values")
            print(f"Unique values: {unique_values}")

    def __augment_minority(self, X, y, seed):
        """
        Augment the minority class using SMOTE (Synthetic Minority Oversampling Technique).
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target labels.
        Returns:
            X (pd.DataFrame), y (pd.Series): Augmented dataset.
        """
        print("[Dataset Status]: Performing data augmentation on the minority class using SMOTE...")
        
        # Convert DataFrame/Series to numpy arrays if necessary
        is_dataframe = isinstance(X, pd.DataFrame)
        X_array = X.values if is_dataframe else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Apply SMOTE
        smote = SMOTE(random_state=seed)
        X_resampled, y_resampled = smote.fit_resample(X_array, y_array)

        # Convert back to DataFrame/Series if input was in that format
        if is_dataframe:
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        print("[Dataset Status]: Augmented dataset size: ", len(X_resampled), "Label ratio: ", sum(y_resampled)/len(y_resampled))
        return X_resampled, y_resampled

    def stratified_split(self, val_size=0.1, test_size=0.2, random_state=42):
        """
        Perform a stratified train-validation-test split.

        Args:
            val_size (float): Proportion of data to use for validation (taken from the training set).
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[Dataset, Optional[Dataset], Optional[Dataset]]: Train, validation, and test datasets.
        """
        X = self.features
        y = pd.Series(self.targets)

        # Split into train and test (if test_size > 0)
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None

        # Split validation from training set (if val_size > 0)
        if val_size > 0:
            val_size_adj = val_size / (1 - test_size)  # Adjust for training set proportion
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size_adj, stratify=y_train, random_state=random_state
            )
        else:
            X_val, y_val = None, None

        # Augment training set if needed
        if self.augment:
            X_train, y_train = self.__augment_minority(X_train, y_train, random_state)

        # Convert to numpy arrays before passing to Dataset class
        train_dataset = TabularDatasetFromArrays(X_train.values, y_train.values, self.cat_column_indices)
        val_dataset = TabularDatasetFromArrays(X_val.values, y_val.values, self.cat_column_indices) if X_val is not None else None
        test_dataset = TabularDatasetFromArrays(X_test.values, y_test.values, self.cat_column_indices) if X_test is not None else None

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[np.ndarray, int]: Features and target label for the sample.
        """
        return self.features.iloc[idx].values, self.targets.iloc[idx]


class TabularDatasetFromArrays(Dataset):
    """
    A simple Dataset wrapper for arrays returned after splitting.
    """
    def __init__(self, features, targets, cat_column_indices):
        self.features = features
        self.targets = targets
        self.cat_column_indices = cat_column_indices

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert data to torch tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)  # Using direct index for numpy arrays
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return features, label


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        """
        Custom DataLoader that passes the cat_column_indices from the dataset.
        This argument needs to be passed to the training function of GAN for the 
        rounding of cat values in data generation.
        
        Args:
            dataset (Dataset): A PyTorch Dataset object that includes the `cat_column_indices` attribute.
            *args, **kwargs: Additional arguments for DataLoader.
        """
        super().__init__(dataset, *args, **kwargs)
        # Store the categorical column indices from the dataset
        self.cat_column_indices = dataset.cat_column_indices

def get_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a DataLoader from a Dataset object.
    
    Args:
        dataset (Dataset): A PyTorch Dataset object (e.g., train_set, val_set, test_set).
        batch_size (int): The size of each batch.
        shuffle (bool): Whether to shuffle the data.
        
    Returns:
        CustomDataLoader: A DataLoader for the given Dataset with categorical column indices included.
    """    
    return CustomDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    


if __name__ == "__main__":
    # Usage example
    # Initialize dataset
    print("[Main]: Initializing dataset...")
    dataset = TabularDataset(
        file_path=FULL_DATA_PATH,
        target_column=TARGET_COLUMN,
        augment=APPLY_AUGMENTATION,
        info=False  # Print dataset info
    )

    print(dataset.cat_column_indices)
    
    # Perform stratified split
    print("[Main]: Performing stratified train-val-test split...")
    train_set, val_set, test_set = dataset.stratified_split(
        val_size=VAL_RATIO, test_size=TEST_RATIO, random_state=SEED
    )

    # Check a few samples from the training set
    print("[Main]: Checking samples from the training set...")
    for i in range(3):  # Check the first 5 samples
        X_sample, y_sample = train_set[i]
        print(f"Sample {i}: Features: {X_sample[:20]}, Label: {y_sample}")

    # Check class distributions
    print("[Main]: Class distributions after split:")
    # Convert tensor labels to Python integers and get the value counts
    train_labels = [train_set[i][1].item() for i in range(len(train_set))]
    val_labels = [val_set[i][1].item() for i in range(len(val_set))] if val_set else None
    test_labels = [test_set[i][1].item() for i in range(len(test_set))] if test_set else None

    # Use pandas to get the value counts for the classes
    print(f"Train ({len(train_labels)} labels): {pd.Series(train_labels).value_counts(normalize=True)}")
    print(f"Validation ({len(val_labels)} labels): {pd.Series(val_labels).value_counts(normalize=True)}") if val_labels else None
    print(f"Test ({len(test_labels)} labels): {pd.Series(test_labels).value_counts(normalize=True)}") if test_labels else None

    # Wrap datasets in DataLoader
    print("[Main]: Creating DataLoaders...")
    train_loader = get_dataloader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_set, batch_size=BATCH_SIZE, shuffle=False) if val_set else None
    test_loader = get_dataloader(test_set, batch_size=BATCH_SIZE, shuffle=False) if test_set else None

    # Print batch samples to confirm
    print("[Main]: Checking a batch from the training DataLoader...")
    for i, (batch_X, batch_y) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"Features shape: {batch_X.shape}, Labels shape: {batch_y.shape}")
        print(f"First batch features:\n{batch_X}")
        print(f"First batch labels:\n{batch_y}")
        break  # Check only the first batch