# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income.csv",na_values=[ " ?"])
```
![o1](<output/Screenshot 2024-10-10 100925.png>)
```
data.isnull().sum()
```
![o2](<output/Screenshot 2024-10-10 100937.png>)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![o3](<output/Screenshot 2024-10-10 100948.png>)
```
data2=data.dropna(axis=0)
data2
```
![o4](<output/Screenshot 2024-10-10 101000.png>)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![o5](<output/Screenshot 2024-10-10 102928.png>)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![o6](<output/Screenshot 2024-10-10 102934.png>)
```
data2
```
![o7](<output/Screenshot 2024-10-10 102940.png>)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![o8](<output/Screenshot 2024-10-10 102946.png>)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![o9](<output/Screenshot 2024-10-10 101042.png>)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![o10](<output/Screenshot 2024-10-10 101051.png>)
```
y=new_data['SalStat'].values
print(y)
```
![o11](<output/Screenshot 2024-10-10 101058.png>)
```
x=new_data[features].values
print(x)
```
![o12](<output/Screenshot 2024-10-10 101103.png>)
```
train_x,test_x,train_y,test_y= train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![o13](<output/Screenshot 2024-10-10 101108.png>)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![o14](<output/Screenshot 2024-10-10 101113.png>)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![o15](<output/Screenshot 2024-10-10 101116.png>)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![o16](<output/Screenshot 2024-10-10 101120.png>)
```
data.shape
```
![o17](<output/Screenshot 2024-10-10 101125.png>)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![018](<output/Screenshot 2024-10-10 101129.png>)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![o19](<output/Screenshot 2024-10-10 101134.png>)
```
tips.time.unique()
```
![o20](<output/Screenshot 2024-10-10 101138.png>)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![o21](<output/Screenshot 2024-10-10 101141.png>)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![o22](<output/Screenshot 2024-10-10 101145.png>)

# RESULT:
  Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.