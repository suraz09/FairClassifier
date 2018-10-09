import sys, os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'preprocess'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'model'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'analysis'))

from preprocess import DataProcessor
from classifier import Classifier
from fairClassifier import FairClassifier
from metrics import FairMetrics

create_gif = False


# Method to parse arguments from command line, also set the threhold to be 70 in this case.
def cli_parser():
    sensitive_attribute = sys.argv[1]
    sensitive_value = sys.argv[2]
    predict_value = 'High'
    predict_column = 'score_text'

    # change the threshold if you want to consider less fairness
    threshold = 70
    return sensitive_attribute, sensitive_value, predict_value, predict_column, threshold



try:
    # parse the attributes from command line
    sensitive_attribute, sensitive_value, predict_value, predict_column, threshold = cli_parser()

    # Load the data from the source
    data = DataProcessor(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'data', 'raw', 'recidivism.csv'))
    # function to pre-process the data
    X,y,Z = data.loadData(sensitive_attribute, sensitive_value, predict_value, predict_column)

    # ##split the data into train and test set
    X_train, X_test, y_train, y_test, Z_train, Z_test = data.split_data(X,y,Z)

    classifier = Classifier()
    # create a nn classifier
    clf = classifier.create_nn_classifier(n_features=X_train.shape[1])
    # train on train set
    history = clf.fit(X_train.values, y_train.values, epochs=20, verbose=0)
    y_pred = pd.Series(clf.predict(X_test).ravel(), index=y_test.index)

    #Result of unfair classifier
    print(f"Accuracy: {100*accuracy_score(y_test, (y_pred>0.5)):.1f}%")

    # Display the result of fairness metric of unfair classifier.
    metrics = FairMetrics()
    print("The classifier satisfies the following %p-rules:")
    print("P-rule {0:.0f}".format(metrics.p_rule(y_pred, Z_test[sensitive_attribute])))

    p_value = metrics.p_rule(y_pred, Z_test[sensitive_attribute])

    if(p_value < threshold):
    # initialise FairClassifier
        clf = FairClassifier(n_features=X_train.shape[1], n_sensitive=Z_train.shape[1], lambdas=[130])

        # pre-train both adverserial and classifier networks
        clf.pretrain(X_train, y_train, Z_train, verbose=0, epochs=5)

        #Get the result of fair classifier
        clf.fit(X_train, y_train, Z_train, validation_data=(X_test, y_test, Z_test), T_iter=160, save_figs=create_gif)
        print("Accuracy ", clf.accuracyArray[len(clf.accuracyArray) -1], "P-rule satisfied ",clf.pruleArray[len(clf.pruleArray)-1])
except Exception as ex:
    print(ex)

#uncomment this line if you you want to visualize accuracy vs fairness tradeoff
#fig = visualize.plotScatter(clf.pruleArray, clf.accuracyArray, x_lab="P-Rule", y_lab="Accuracy")
