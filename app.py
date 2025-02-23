from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
with open('aggragaterating.pkl','rb') as model_files:
  rating=pickle.load(model_files)
@app.route('/')
def home():
  return render_template('data.html')
@app.route('/read' ,methods=['POST'])
def model():
  Restaurant_Name=request.form['Restaurant Name']
  Country_Code=int(request.form['Country Code'])
  City=request.form['City']
  Address=request.form['Address']
  Locality_Verbose=request.form['Locality Verbose']
  Longitude=float(request.form['Longitude'])
  Latitude=float(request.form['Latitude'])
  Cuisines=request.form['Cuisines']
  Average_Cost_for_two=int(request.form['Average Cost for two'])
  Currency=(request.form['Currency'])
  Has_Table_booking=(request.form['Has Table booking'])
  Has_Online_delivery=(request.form['Has Online delivery'])
  Is_delivering_now=(request.form['Is delivering now'])
  Price_range=float(request.form['Price range'])
  Votes=int(request.form['Votes'])
  Rating_text=request.form['Rating text']
  data=pd.DataFrame({
    'Restaurant_Name':[Restaurant_Name],
    'Country_Code':[Country_Code],
    'City':[City],
    'Address':[Address],
    'Locality_Verbose':[Locality_Verbose],
    'Longitude':[Longitude],
    'Latitude':[Latitude],
    'Cuisines':[Cuisines],
    'Average_Cost_for_two':[Average_Cost_for_two],
    'Currency':[Currency],
    'Has_Table_booking':[Has_Table_booking],
    'Has_Online_delivery':[Has_Online_delivery],
    'Is_delivering_now':[Is_delivering_now],
    'Price_range':[Price_range],
    'Rating_text':[Rating_text],
    'Votes':[Votes]
  })
  rating_mapping = {
    'Excellent': 5,
    'Very Good': 4,
    'Good': 3,
    'Average': 2,
    'Poor': 1,
    'Not rated': 0
  }
  data['Rating'] = data['Rating_text'].map(rating_mapping)
  data.drop(columns=['Rating_text'],axis=1,inplace=True)
  data['Name_Length'] = data['Restaurant_Name'].apply(len)
  data['Address_Length'] = data['Address'].apply(len)
  status_mapping = {
    'No': 0,
    'Yes': 1
  }
  data['Has_Table_booking']=data['Has_Table_booking'].map(status_mapping)
  data['Has_Online_delivery']=data['Has_Online_delivery'].map(status_mapping)
  data['Is_delivering_now']=data['Is_delivering_now'].map(status_mapping)
  data.drop(columns=['Restaurant_Name'],axis=1,inplace=True)
  categorical_col=['Currency','Cuisines','Locality_Verbose','Address','City']
  for col in categorical_col:
   frequency_encoding = data[col].value_counts()
   data[col] = data[col].map(frequency_encoding)
  data.rename(columns={
    'Country_Code': 'Country Code',
    'Locality_Verbose': 'Locality Verbose',
    'Average_Cost_for_two': 'Average Cost for two',
    'Has_Table_booking': 'Has Table booking',
    'Has_Online_delivery': 'Has Online delivery',
    'Is_delivering_now': 'Is delivering now',
    'Price_range':'Price range'

  }, inplace=True)

  prediction=rating.predict(data)
  return render_template('output.html',prediction_result=prediction[0])
if __name__=='__main__':
  app.run(debug=True)

  
