import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import xgboost as xgb

def load_data(file_path):
    """
    Load the preprocessed data from a CSV file.
    """
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)
    
def train_and_evaluate(x_train, x_test, y_train, y_test):
    """
    Train the XGBoost Classifier and evaluate its performance.
    """
    print("Unique values in y_train:",np.unique(y_train))
    print("Unique values in y_test:",np.unique(y_test))
    
    #Compute the class weights
    class_weights=compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Create a weight array for the training set
    weights = np.array([class_weight_dict[label] for label in y_train])

    # Initialize the XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=class_weights[1] / class_weights[0]  # Handle class imbalance
    )

    # Train the model with sample weights
    model.fit(x_train, y_train, sample_weight=weights)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate model performance
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
    
def save_model(model,filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")
    

if __name__=="__main__":
    # File paths
    preprocessed_file_path = 'preprocessed_cic_ids2017_10percent_classifier.csv'
    model_filename = 'nids_xgboost_classifier_model.pkl'

    # Load the preprocessed data
    df = load_data(preprocessed_file_path)

    # Import split_data function from the classifier preprocessing script
    from preprocessing_classifier import split_data

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = split_data(df)

    # Train and evaluate the classifier model
    model = train_and_evaluate(x_train, x_test, y_train, y_test)

    # Save the trained model
    save_model(model, model_filename)	
    
    
    
    
