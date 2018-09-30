import sys, os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'preprocess'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'model'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'analysis'))

from preprocess import DataProcessor
from classifier import Classifier
import visualize

#Load the data from the source
data = DataProcessor(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'data', 'raw', 'recidivism.csv'))

#calls the function to process the data
X,y,Z = data.loadProcessData()

#split the data into train and test set
X_train, X_test, y_train, y_test, Z_train, Z_test = data.split_data(X,y,Z)

#Load the classifier
classifier = Classifier()

#create a nn classifier
clf = classifier.create_nn_classifier(n_features=X_train.shape[1])

# train on train set
history = clf.fit(X_train.values, y_train.values, epochs=20, verbose=0)

y_pred = pd.Series(clf.predict(X_test).ravel(), index=y_test.index)
print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")
print(f"Accuracy: {100*accuracy_score(y_test, (y_pred>0.5)):.1f}%")

#visualize the distribution of the result and save the image as it as biased_training
fig = visualize.plot_distributions(y_pred, Z_test, fname='biased_training.png')