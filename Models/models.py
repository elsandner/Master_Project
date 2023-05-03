
# fit an LSTM network to training data
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from keras import backend as K
from keras.utils import register_keras_serializable


# LSTM:
def lstm_0(train_X, train_y):

    #design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(train_X.shape[2])) # Predict all features!
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

    return model

#Asked ChatGPT to improve lstm_0
def lstm_1(train_X, train_y):
    # design network
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(train_X.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=64, verbose=1, shuffle=False, validation_split=0.1)

    return model

def lstm_2(train_X, train_y):
    # design network
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dense(train_X.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=64, verbose=1, shuffle=False, validation_split=0.1)

    return model

# PINN:
def pinn_0(train_X, train_y, alpha):

    def custom_loss():
        def loss(y_true, y_pred):
            # Split y_true and y_pred into two features
            y_true_f1, y_true_f2 = tf.split(y_true, num_or_size_splits=2, axis=1)
            y_pred_f1, y_pred_f2 = tf.split(y_pred, num_or_size_splits=2, axis=1)

            # Calculate the mean squared error for each feature
            mse_f1 = K.mean(K.square(y_true_f1 - y_pred_f1), axis=-1)
            mse_f2 = K.mean(K.square(y_true_f2 - y_pred_f2), axis=-1)

            # Calculate the weighted loss
            weighted_loss = 0.5 * mse_f1 + (1 - 0.5) * mse_f2

            return weighted_loss

        return loss

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(train_X.shape[2]))  # Predict all features!

    model.compile(optimizer='adam',
                  loss=custom_loss(),
                  )

    # fit network
    model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

    return model

def pinn_1(train_X, train_y, alpha):

    def custom_loss():
        def loss(y_true, y_pred):
            # Split y_true and y_pred into two features
            y_true_f1, y_true_f2 = tf.split(y_true, num_or_size_splits=2, axis=1)
            y_pred_f1, y_pred_f2 = tf.split(y_pred, num_or_size_splits=2, axis=1)

            # Calculate the mean squared error for each feature
            mse_f1 = K.mean(K.square(y_true_f1 - y_pred_f1), axis=-1)
            mse_f2 = K.mean(K.square(y_true_f2 - y_pred_f2), axis=-1)

            # Calculate the weighted loss
            weighted_loss = 0.5 * mse_f1 + (1 - 0.5) * mse_f2

            return weighted_loss

        return loss

    # design network
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(train_X.shape[2]))

    model.compile(optimizer='adam',
                  loss=custom_loss(),
                  )

    # fit network
    model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

    return model
# ---------------------------------

model_dictionary = {
    #LSTM:
    "lstm_0": lstm_0,
    "lstm_1": lstm_1,
    "lstm_2": lstm_2,

    #PINN:
    "pinn_0": pinn_0,
    "pinn_1": pinn_1
    #....
}


def get_model(model_name, train_X, train_y, alpha=None):

    if model_name.startswith("pinn"):
        model_function = model_dictionary[model_name]
        return model_function(train_X, train_y, alpha)
    else:
        model_function = model_dictionary[model_name]
        return model_function(train_X, train_y)



