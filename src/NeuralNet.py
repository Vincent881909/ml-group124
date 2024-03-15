# Imports
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('../data/train_enc.csv')

train, validation = train_test_split(df, test_size=0.2, random_state=200, shuffle=True)

target = 'SalePrice'
X_train = train.drop(target, axis=1)
y_train = train[target]

X_validation = validation.drop(target, axis=1)
y_validation = validation[target]

# Build the model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.keras'
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', save_best_only=True, mode='auto')
# callbacks_list = [checkpoint]

# Train the model
history = NN_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

# Record predictions for the test set
test = pd.read_csv('../data/test_enc.csv')
predictions = NN_model.predict(test)

# add the Id column to the predictions
df_test = pd.read_csv('../data/test.csv')
test_predictions = pd.DataFrame({'Id': df_test['Id']})
test_predictions['SalePrice'] = predictions

# Save the predictions to a CSV file
test_predictions.to_csv('../nn_test_predictions_1.csv', index=False)
