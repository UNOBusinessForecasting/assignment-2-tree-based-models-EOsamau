# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load training data
TrainData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')

# Prepare training data
Y = TrainData['meal']
X = TrainData.drop(columns=['meal', 'id', 'DateTime'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Create and train the model
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
modelFit = model.fit(x_train, y_train)

# Predictions and accuracy
pred_train = model.predict(x_train)
pred_test = model.predict(x_test)

train_accuracy = 100 * accuracy_score(y_train, pred_train)
test_accuracy = 100 * accuracy_score(y_test, pred_test)

print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

# Check accuracy threshold (e.g., 70%)
assert test_accuracy > 70, "Test accuracy is below the expected threshold!"

# Load and predict on test data
TestData = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
xt = TestData.drop(columns=['meal', 'id', 'DateTime'], axis=1)
pred = model.predict(xt)
print(pred)

# Ensure prediction size matches the input
assert len(pred) == len(xt), "Prediction length does not match input length!"
