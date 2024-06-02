import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    poly = PolynomialFeatures(degree=2, include_bias=True)
    pkl_file = open('genre_encoder.pkl', 'rb')
    le_genre = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('namecategory_encoder.pkl', 'rb')
    le_namecategory = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('platform_encoder.pkl', 'rb')
    le_platform = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('publisher_encoder.pkl', 'rb')
    le_publisher = pickle.load(pkl_file)
    pkl_file.close()

    platform = request.form['platform']
    year = request.form['year']
    genre = request.form['genre']
    publisher = request.form['publisher']
    eu_sales = request.form['sales']
    name_category = request.form['name_category']

    data = np.array([platform, year, genre, publisher, eu_sales, name_category])
    df = pd.DataFrame(data.reshape(1, -1), columns=['Platform','Year', 'Genre', 'Publisher','EU_Sales','Name_category'])
    df['Platform'] = le_platform.fit_transform(df['Platform'])
    df['Genre'] = le_genre.fit_transform(df['Genre'])
    df['Publisher'] = le_publisher.fit_transform(df['Publisher'])
    df['Name_category'] = le_namecategory.fit_transform(df['Name_category'])
    
    
    data_poly = poly.fit_transform(df)
    prediction = model.predict(data_poly)
    final = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Global Sales: $ {} Million'.format(final))

if __name__ == "__main__":
    app.run(debug=True)
