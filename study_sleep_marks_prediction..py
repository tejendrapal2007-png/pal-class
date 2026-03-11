import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data=pd.DataFrame({
    'Study_Hours':[1,2,3,4],
    'Sleep_Hours':[5,6,7,8],
    'Marks':[50,60,70,85]
})
x=data[['Study_Hours','Sleep_Hours']]
y=data['Marks']

poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
X_interaction=poly.fit_transform(x)

columns=poly.get_feature_names_out(x.columns)
df_interaction=pd.DataFrame(X_interaction,columns=columns)

print("Interaction Data:")
print(df_interaction)

model=LinearRegression()
model.fit(X_interaction,y)

print("\nCoeffiecients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

predicted_marks=model.predict(X_interaction)
print("\nPredicted Marks:")
print(predicted_marks)