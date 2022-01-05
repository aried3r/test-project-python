import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATASET_URL = '' # set the url or local reference
FEATURE_NAME = '' # set target feature name
TEST_SIZE = 0.2

def load_data():
    data = pd.read_csv(DATASET_URL, sep=';') # implement the custom logic here
    return data

# load the data and separate out the target feature data
data = load_data()
y = data[FEATURE_NAME]
X = data.drop(FEATURE_NAME, axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=123, stratify=y)

# Create LogisticRegression object and train
reg = LogisticRegression()
reg.fit(X_train, y_train)

# Make predictions using the testing set
y_predict = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
