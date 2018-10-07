from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'analysis'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'configs' ))

from metrics import FairMetrics
import constant

"""
FairClassifier equipped with adversarial network
"""
class FairClassifier(object):

    def __init__(self, n_features, n_sensitive, lambdas):
        self.lambdas = lambdas

        clf_inputs = Input(shape=(n_features,))
        adv_inputs = Input(shape=(1,))
        self.metrics = FairMetrics()

        clf_net = self._create_clf_net(clf_inputs)
        adv_net = self._create_adv_net(adv_inputs, n_sensitive)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(clf_inputs, clf_net, adv_net, n_sensitive)
        self._val_metrics = None
        self._fairness_metrics = None

        self.predict = self._clf.predict
        self.accuracyArray = []
        self.pruleArray = []

    """
    Make the layers of the classifier trainable
    """
    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable
    """
    Create a 3 layer Neural Network
    """
    def _create_clf_net(self, inputs):
        dense1 = Dense(constant.UNIT, activation='relu')(inputs)
        dropout1 = Dropout(constant.DROP_RATE)(dense1)
        dense2 = Dense(constant.UNIT, activation='relu')(dropout1)
        dropout2 = Dropout(constant.DROP_RATE)(dense2)
        dense3 = Dense(constant.UNIT, activation='relu')(dropout2)
        dropout3 = Dropout(constant.DROP_RATE)(dense3)
        outputs = Dense(constant.OUTPUT_UNIT, activation='sigmoid', name='y')(dropout3)
        return Model(inputs=[inputs], outputs=[outputs])

    """
    Create a 3 layer adversarial Neural Network
    """
    def _create_adv_net(self, inputs, n_sensitive):
        dense1 = Dense(constant.UNIT, activation='relu')(inputs)
        dense2 = Dense(constant.UNIT, activation='relu')(dense1)
        dense3 = Dense(constant.UNIT, activation='relu')(dense2)
        outputs = [Dense(constant.OUTPUT_UNIT, activation='sigmoid')(dense3) for _ in range(n_sensitive)]
        return Model(inputs=[inputs], outputs=outputs)

    """
    Compile the classifier Neural Network
    """
    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        return clf

    """
    Compile classifier with adversarial network
    """
    def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs), adv_net(clf_net(inputs))])
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.] + [-lambda_param for lambda_param in self.lambdas]
        clf_w_adv.compile(loss=['binary_crossentropy'] * (len(loss_weights)),
                          loss_weights=loss_weights,
                          optimizer='adam')
        return clf_w_adv

    """
    Compile adversarial Network
    """
    def _compile_adv(self, inputs, clf_net, adv_net, n_sensitive):
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        adv.compile(loss=['binary_crossentropy'] * n_sensitive, optimizer='adam')
        return adv

    """
    Compute class weights of the training set
    """
    def _compute_class_weights(self, data_set):
        class_values = [0, 1]
        class_weights = []
        if len(data_set.shape) == 1:
            balanced_weights = compute_class_weight('balanced', class_values, data_set)
            class_weights.append(dict(zip(class_values, balanced_weights)))
        else:
            n_attr = data_set.shape[1]
            for attr_idx in range(n_attr):
                balanced_weights = compute_class_weight('balanced', class_values,
                                                        np.array(data_set)[:, attr_idx])
                class_weights.append(dict(zip(class_values, balanced_weights)))
        return class_weights

    """
    Compute class weight of the target set
    """
    def _compute_target_class_weights(self, y):
        class_values = [0, 1]
        balanced_weights = compute_class_weight('balanced', class_values, y)
        class_weights = {'y': dict(zip(class_values, balanced_weights))}
        return class_weights

    """
    Pretrain Classifier and adversarial network on initial data set
    """
    def pretrain(self, x, y, z, epochs=10, verbose=0):
        self._trainable_clf_net(True)
        self._clf.fit(x.values, y.values, epochs=epochs, verbose=verbose)
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        class_weight_adv = self._compute_class_weights(z)
        self._adv.fit(x.values, np.hsplit(z.values, z.shape[1]), class_weight=class_weight_adv,
                      epochs=epochs, verbose=verbose)

    """
    Train and test the accuracy and fairness of the model
    """
    def fit(self, x, y, z, validation_data=None, T_iter=250, batch_size=128,
            save_figs=False):
        n_sensitive = z.shape[1]
        if validation_data is not None:
            x_val, y_val, z_val = validation_data

        class_weight_adv = self._compute_class_weights(z)
        class_weight_clf_w_adv = [{0: 1., 1: 1.}] + class_weight_adv
        self._val_metrics = pd.DataFrame()
        self._fairness_metrics = pd.DataFrame()
        for idx in range(T_iter):
            if validation_data is not None:
                y_pred = pd.Series(self._clf.predict(x_val).ravel(), index=y_val.index)
                self.accuracyArray.append(accuracy_score(y_val, (y_pred > 0.5)) * 100)
                # uncomment this line if you want to see the accuracy score
                # print("Accuracy ", accuracy_score(y_val, (y_pred > 0.5)) * 100)
                for sensitive_attr in z_val.columns:
                    self.pruleArray.append(self.metrics.p_rule(y_pred, z_val[sensitive_attr]))
                    # uncomment this line if you want to see the fairness metrics
                    # print("P-rule ", self.metrics.p_rule(y_pred, z_val[sensitive_attr]))

            # train adverserial
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            self._adv.fit(x.values, np.hsplit(z.values, z.shape[1]), batch_size=batch_size,
                          class_weight=class_weight_adv, epochs=1, verbose=0)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            self._clf_w_adv.train_on_batch(x.values[indices],
                                           [y.values[indices]] + np.hsplit(z.values[indices], n_sensitive),
                                           class_weight=class_weight_clf_w_adv)