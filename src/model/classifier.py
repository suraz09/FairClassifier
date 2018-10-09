from keras.layers import Input, Dense, Dropout
from keras.models import Model
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'configs' ))
import constant

"""
Classifier class contains method to create NN classifier
"""
class Classifier:

    def __init__(self):
        pass

    """
    Create a NN classifier given the shape number of features as input tensor.
    Returns the created model
    """
    def create_nn_classifier(self,n_features):
        inputs = Input(shape=(n_features,))
        dense1 = Dense(constant.UNIT, activation='relu')(inputs)
        dropout1 = Dropout(constant.DROP_RATE)(dense1)
        dense2 = Dense(constant.UNIT, activation='relu')(dropout1)
        dropout2 = Dropout(constant.DROP_RATE)(dense2)
        dense3 = Dense(constant.UNIT, activation="relu")(dropout2)
        dropout3 = Dropout(constant.DROP_RATE)(dense3)
        outputs = Dense(constant.OUTPUT_UNIT, activation='sigmoid')(dropout3)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model
