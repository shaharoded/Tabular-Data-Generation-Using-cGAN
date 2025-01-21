import pandas as pd
import numpy as np
from scipy.io import arff
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


# Local Code
from config import *


class TabularDataset(Dataset):
    """
    A PyTorch Dataset class for tabular data handling, including preprocessing,
    augmentation, and stratified train-validation-test splitting.

    Attributes:
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

        # Load ARFF file
        self.data, self.meta = arff.loadarff(file_path)
        self.df = pd.DataFrame(self.data)

        # Display initial dataset information if requested
        if info:
            self.__print_info()

        # Preprocess the data (normalization and encoding)
        self.__preprocess()

    def __print_info(self):
        """
        Print initial dataset information such as row/column counts,
        target class distribution, and column details.
        """
        print("[Info]: Initial Dataset Summary")

        # Number of rows and columns
        num_rows, num_cols = self.df.shape
        print(f"[Info]: Number of rows: {num_rows}")
        print(f"[Info]: Number of columns: {num_cols}")

        # Target class distribution
        target_dist = self.df[self.target_column].value_counts(normalize=True)
        print(f"[Info]: Target class distribution:\n{target_dist}\n")

        # Null values in columns
        null_values = self.df.isnull().sum()
        print(f"[Info]: Null values in each column:\n{null_values}\n")

        # One-hot encoded column count
        df_encoded = pd.get_dummies(self.df, drop_first=True)
        print(f"[Info]: Number of columns after one-hot encoding: {df_encoded.shape[1]}")

    def __preprocess(self):
        """
        Preprocess the dataset:
        - Normalize continuous features using MinMax scaling.
        - One-hot encode categorical features.
        
        All vector inputs are normalized to [-1,1] range to ensure consistency.
        """
        print("[Dataset Status]: Preprocessing dataset...")

        # Separate target column and features
        self.targets = self.df[self.target_column].astype("category").cat.codes
        
        features = self.df.drop(columns=[self.target_column])

        # Normalize continuous features
        cont_cols = features.select_dtypes(include=["number"]).columns
        scaler = MinMaxScaler(feature_range=(-1, 1))  # Modify the range to [-1, 1]
        features[cont_cols] = scaler.fit_transform(features[cont_cols])

        # One-hot encode categorical features
        cat_cols = features.select_dtypes(include=["object", "category"]).columns
        cat_encoded = pd.get_dummies(features[cat_cols], drop_first=True).astype(int)
        cat_encoded = cat_encoded * 2 - 1 # Modify the range to [-1, 1] 
        
        # Create a list of indices for all one-hot encoded columns
        self.cat_column_indices = list(range(len(cont_cols), len(cont_cols) + len(cat_encoded.columns)))

        # Combine normalized continuous and one-hot encoded categorical features
        features = pd.concat([features[cont_cols], cat_encoded], axis=1)

        # Update the features attribute
        self.features = features
        self.num_features = features.shape[1]

    def __augment_minority(self, X, y):
        """
        Augment the minority class with randomized noise for each augmented record.
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target labels.
        Returns:
            X (pd.DataFrame), y (pd.Series): Augmented dataset.
        """
        print("[Dataset Status]: Performing data augmentation on the minority class...")
        
        class_counts = y.value_counts()
        max_count = class_counts.max()

        augmented_features = []
        augmented_labels = []

        for label, count in class_counts.items():
            if count < max_count:
                diff = max_count - count
                class_indices = y[y == label].index
                
                # Oversample from minority class
                oversampled_indices = np.random.choice(class_indices, diff, replace=True)
                
                # Generate randomized noise for each record
                for idx in oversampled_indices:
                    record = X.loc[idx]  # Original record
                    noise = np.random.normal(-0.05, 0.05, size=record.shape)  # Random noise
                    augmented_features.append(record + noise)
                    augmented_labels.append(label)

        # Append augmented data to original dataset
        if augmented_features:
            augmented_features = pd.DataFrame(augmented_features, columns=X.columns)
            augmented_labels = pd.Series(augmented_labels, name=y.name)
            X = pd.concat([X, augmented_features], axis=0, ignore_index=True)
            y = pd.concat([y, augmented_labels], axis=0, ignore_index=True)

        return X, y

    def stratified_split(self, val_size=0.1, test_size=0.2, random_state=None):
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
            X_train, y_train = self.__augment_minority(X_train, y_train)

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

    # Perform stratified split
    print("[Main]: Performing stratified train-val-test split...")
    train_set, val_set, test_set = dataset.stratified_split(
        val_size=VAL_RATIO, test_size=TEST_RATIO, random_state=SEED
    )

    # Check a few samples from the training set
    print("[Main]: Checking samples from the training set...")
    for i in range(3):  # Check the first 5 samples
        X_sample, y_sample = train_set[i]
        print(f"Sample {i}: Features: {X_sample[:8]}, Label: {y_sample}")

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