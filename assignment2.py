# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# %%
TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
TrainData.head()

# %%
from sklearn.model_selection import train_test_split

Y = TrainData['meal']
X = TrainData.drop(columns=['meal','id','DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# %%
model = DecisionTreeClassifier(max_depth = 200, min_samples_leaf= 5)

modelFit = model.fit(x_train,y_train )

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

pred_train = model.predict(x_train)
pred_test = model.predict(x_test)


print(100*accuracy_score(y_train, model.predict(x_train)))


print(100*accuracy_score(y_test, model.predict(x_test)))


# %%
TestData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
xt = TestData.drop(columns = ['meal','id','DateTime'], axis =1)
pred = model.predict(xt)

# %%



