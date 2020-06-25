import numpy as np
# converts text file to array
from numpy import genfromtxt
# automatically shuffles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

data = genfromtxt('../data/bank_note_data.txt', delimiter=',')
# labels
y = data[:,4]
# features
X = data[:,0:4]
# random state ensures same randomized test set each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# standardize data
scaler_object = MinMaxScaler()
# fit to training data only, then transform both train and test data
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)
