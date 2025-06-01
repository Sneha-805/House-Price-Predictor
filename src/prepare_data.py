import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), '../data/train.csv')
df=pd.read_csv(file_path)
new_df=df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].copy()
new_df.columns=['square_footage', 'bedrooms', 'bathrooms', 'price']
output_path = os.path.join(os.path.dirname(__file__), '../data/house_price_data.csv')
new_df.to_csv(output_path,index=False)
print("simplified dataset is ready")
