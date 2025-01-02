import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """	
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)
    

def clean_data(df):
    """
    Clean the dataset by handling missing values and duplicates.
    """
    df=df.dropna() #Drop rows with missing values
    df=df.drop_duplicates() #Remove Duplicate rows
    print(f"Data cleaned: {df.shape[0]} rows and {df.shape[1]} columns.")
    return df
    
    
def feature_engineering(df):
    """
    Perform feature engineering, such as scaling the numerical features
    and encoding categorical features.
    """
    # Handle categorical columns
    # If there's a column 'Label' or any categorical column, you can encode it.
    # This assumes 'Label' is the target column and needs encoding (you can modify this part).
    if 'Label' in df.columns:
          le = LabelEncoder()
          df['Label'] = le.fit_transform(df['Label'])
    
    # Identify numerical columns to scale
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Handle infinite values by replacing them with NaN
    df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with the mean of each column
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    
    # Create an instance of the scaler for numerical data
    scaler = StandardScaler()

    # Scale the numerical columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    print("Feature engineering completed (data scaled and categorical data encoded).")
    return df
    
    
def split_data(df):
    """
    Split the data into features (X) and target (y) and then into training and testing sets.
    """
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces

    if 'Label' not in df.columns:
    	 raise KeyError("The 'Label' column is not present in dataset.")
    	 
    x=df.drop('Label',axis=1) #Features (excluding Label)
    y=df['Label'] # Label (target variable - attack or anomaly)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    print(f"Data split into train and test sets: {x_train.shape[0]} train samples , {x_test.shape[0]} test samples.")
    return x_train,x_test,y_train,y_test
    
    
def save_preprocessed_data(df,file_path):
    """
    Save the preprocessed data to a new CSV file.
    """
    df.to_csv(file_path,index=False) 
    print(f"Preprocessed data saved to {file_path}.")
    
    
if __name__=="__main__":
    #Update the path of the combined csv file to combined file path
    combined_file_path='combined_cic_ids2017_10percent.csv'
    preprocessed_file_path='preprocessed_cic_ids2017_10percent.csv'
    
    # Step 1: Load the Data
    df=load_data(combined_file_path)
    
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces
    
    #Step 2: Clean the data
    df=clean_data(df)
    
    #Step 3: Feature Engineering (scaling)
    df=feature_engineering(df)
    
    print("Columns in the DataFrame:", df.columns)
    
    if 'Label' in df.columns:
    	x = df.drop('Label', axis=1)  # Features (excluding Label)
    	y = df['Label']  # Label (target variable)
    else:
    	print("The 'Label' column is missing!")
    	exit()
    
    print("Class distribution before split:")
    print(y.value_counts())
    
    #Step 4:Split the data into train and test data
    x_train,x_test,y_train,y_test=split_data(df)
    
    print("\nClass Distribution in the training set:")
    print(y_train.value_counts())
    
    print("\nClass Distribution in the testing set:")
    print(y_test.value_counts())
    
    #Step 5: Save preprocessed data for further use in project
    save_preprocessed_data(df,preprocessed_file_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
