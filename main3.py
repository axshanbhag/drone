from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import pandas as pd

df = pd.read_csv('titanic.csv')
le = LabelEncoder()
# encode sex into numerical values
df['Sex_encoded'] = le.fit_transform(df['Sex'])
for col in ['Age', 'Fare']:
    if df[col].isnull().any():
        df[col].fillna(df[col].median())
X = df[['Age', 'Fare', 'Sex_encoded']]  # features are the inputs
y = df['Survived']  # target is what the model is trying to predict

# random ensures that we get the same split every time we run the code
# test_size is the proportion of the dataset to include in the test split 20% here
# stratify ensures that the ratio of people that died and survived are consistent in training/testing sets
# splits dataset into training and testing sets depending on features and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2, stratify = y)
# create and train the model
clf = DecisionTreeClassifier(random_state=42)
# fits the model to the training data by lookiing for patterns in relation to the target
# ex: female + higher fare = more likely to survive
clf.fit(X_train, y_train)
# predict on the test set
#y_pred are the predicted values from the X_test set values
y_pred = clf.predict(X_test)
# compares model's predictions to the actual values in a fraction of correct predictions
print("Accuracy:", accuracy_score(y_test, y_pred))
# confusion matrix and classification report to see how well the model did
cm = confusion_matrix(y_test, y_pred) 
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred))
# rows 0, 1 are classes of who died/survived
# columns are what the model predicted
# precision is out of what the model predicted, how many were actually correct
# how often is it right when it says this person survived/died
# recall is out of what actually happened, how many did the model predict correctly
# how many real survivors did it find
# f1-score is a weighted average of precision and recall
# support is the number of actual occurrences of the class in the specified dataset
ConfusionMatrixDisplay(cm, display_labels=['Died (0)', 'Survived (1)']).plot()
plt.title("Confusion Matrix (baseline DecisionTree)")
#plt.show()


# tuning max_depth parameter to see if it improves accuracy
# right now the max_depth is unlimited which can cause the model to overfit
acc_max_depth = 0
for depth in range(1, 16): #depths from 1 to 15
    clf_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf_depth.fit(X_train, y_train)
    y_pred_depth = clf_depth.predict(X_test)
    acc_depth = accuracy_score(y_test, y_pred_depth)
    print("max_depth=", str(depth), "-> test accuracy=", str(acc_depth))
    if acc_depth > acc_max_depth:
        acc_max_depth = acc_depth
        best_depth = depth

print("Best max_depth:", best_depth, "with accuracy:", acc_max_depth)

final_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_clf.fit(X_train, y_train)
final_y_pred = final_clf.predict(X_test)
print("Final model accuracy:", accuracy_score(y_test, final_y_pred))
cm_final = confusion_matrix(y_test, final_y_pred)
print("Final confusion matrix:\n", cm_final)
