import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import r2_score,mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
import pickle

df = pd.read_csv('vgsales.csv')

df['Name_category'] = ''

def adjust(row):
    a = row['Name'].lower().split(' ')
    if (a[0]=='just') or (a[0]=='street') or (a[0]=='brian') or (a[0]=='gran') or (a[0]=="assassin's")or (a[0]=="the") :
            row['Name_category'] = str.join(' ',[row['Name_category'],a[0],a[1]])
    elif (a[0]=='call') or (a[0]=='metal') or (a[0]=='need') or (a[0]=='lego') or (a[0]=="god") :
            row['Name_category'] = str.join(' ',[row['Name_category'],a[0],a[1]])
    else :
            row['Name_category'] = str.join(' ',[row['Name_category'],a[0]])
   
    return row['Name_category']

for i in range (df.shape[0]):
    df.iloc[i,-1]=adjust(df.iloc[i])  

df = df.drop('Name',axis= 1)
df = df.dropna()

le = preprocessing.LabelEncoder()

lst = ['Name_category','Platform','Genre','Publisher']
df['Name_category'] = le.fit_transform(df['Name_category'])


output1 = open('namecategory_encoder.pkl', 'wb')
pickle.dump(le, output1)
df['Platform'] = le.fit_transform(df['Platform'])
output2 = open('platform_encoder.pkl', 'wb')
pickle.dump(le, output2)
df['Genre'] = le.fit_transform(df['Genre'])
output3 = open('genre_encoder.pkl', 'wb')
pickle.dump(le, output3)
df['Publisher'] = le.fit_transform(df['Publisher'])
output4 = open('publisher_encoder.pkl', 'wb')
pickle.dump(le, output4)

df_2=pd.DataFrame()
lst = ['Platform','Year', 'Genre', 'Publisher','EU_Sales','Name_category']
for i in lst :
    df_2[i]=df[i]

X = df_2.values
Y = df['Global_Sales'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)
polynomial = LinearRegression()
polynomial.fit(x_train_trans, y_train)

pickle.dump(polynomial, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))