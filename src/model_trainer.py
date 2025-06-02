import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import joblib

data_path=os.path.join(os.path.dirname(__file__), "../data/house_price_data.csv")
df=pd.read_csv(data_path)
X=df[['square_footage', 'bedrooms', 'bathrooms']]
y=df['price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
model=LinearRegression()
model.fit(X_train,y_train)

model_path=os.path.join(os.path.dirname(__file__),"../models/house_price_model.pkl")
joblib.dump(model,model_path)
print("model saved successfully")