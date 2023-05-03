
# fit an LSTM network to training data
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout


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


model_dictionary = {
    "lstm_0": lstm_0,
    "lstm_1": lstm_1,
    #....
}


def get_model(model_name, train_X, train_y):
    model_function = model_dictionary[model_name]
    return model_function(train_X, train_y)



