from keras.layers import Input, Dense, Dropout
from keras.models import Model

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
        dense1 = Dense(32, activation='relu')(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation="relu")(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation='sigmoid')(dropout3)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model