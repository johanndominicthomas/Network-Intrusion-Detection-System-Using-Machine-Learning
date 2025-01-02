import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb

def load_data(file_path):
	print(f"Loading data from {file_path}...")
	return pd.read_csv(file_path)
	
def train_and_evaluate(x_train,x_test,y_train,y_test):
	print("Unique values in y_train:",np.unique(y_train))
	print("Unique values in y_test:",np.unique(y_test))
	
	
	#Handle class weights by computing class weights
	
	
	#Initialize the XGBoost model
	model=xgb.XGBRegressor(
		n_estimators=100,
		random_state=42,
		
	)
	
	#Train the model
	model.fit(x_train,y_train)
	
	#Make predictions on the test set
	y_pred=model.predict(x_test)
	
	
	
	#Evaluate the model performance
	print("\nModel Evaluation:")
	print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
	print("R² Score:", r2_score(y_test, y_pred))
	
	return model
	
#Save the trainedd model
def save_model(model,filename):
	joblib.dump(model,filename)
	print(f"\nModel saved to {filename}")
	
#Main script execution
if __name__=="__main__":
	#Define file paths
	preprocessed_file_path='preprocessed_cic_ids2017_10percent.csv'
	model_filename='nids_xgboost_regressor_model.pkl' #Trained model file name
	
	#Load preprocessed data
	df=load_data(preprocessed_file_path)
	
	#Assuming data is already split in preprocessing.py or else splitting to be done
	from preprocessing import split_data
	x_train,x_test,y_train,y_test=split_data(df)
	
	#Train and evaluate model
	model=train_and_evaluate(x_train,x_test,y_train,y_test)
	
	#Save trained model for later use
	save_model(model,model_filename)

















	
	
	
	
