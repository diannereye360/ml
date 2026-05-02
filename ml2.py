from sklearn.datasets import fetch_california_housing
c = fetch_california_housing()
print(c.data.shape)
print(c.target.shape)
print(c.feature_names)

import pandas as pd

pd.set_option("display.precision",2)
pd.set_option("display.max_columns", 9)
pd.set_option("display.width", None)

cali_df = pd.DataFrame(c.data, columns=c.feature_names)
cali_df['MedHouseVal'] = pd.Series(c.target)
print(cali_df.head())

print(cali_df.describe())

sample_df = cali_df.sample(frac=0.1, random_state=17)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("whitegrid") 

for feature in c.feature_names:
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample_df, 
        x=feature, 
        y='MedHouseVal', 
        hue='MedHouseVal',
        palette="cool",
        legend=False
        )
    
    #plt.show()

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        c.data, c.target, random_state=11)
    
    print(x_train.shape)
    print(y_train.shape)    
    print(x_test.shape)
    print(y_test.shape) 

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X=x_train, y=y_train)
    
    for i, name in enumerate(c.feature_names):
        print(f"{name}: {lr.coef_[i]:.3f}")

    predicted = lr.predict(X=x_test)
    expected = y_test
   
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

import matplotlib.pyplot as plt2
figure = plt2.figure(figsize=(9,9))
axes = sns.scatterplot(data=df, 
                       x='Expected', 
                       y='Predicted', 
                       hue='Expected', 
                       palette="cool", 
                       legend=False)

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
axes.set_xlim(start, end)   
axes.set_ylim(start, end)

line = plt2.plot([start, end], [start, end], 'k--')
plt2.show()
