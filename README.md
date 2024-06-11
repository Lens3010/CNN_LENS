
X = train.drop(columns='Survived').to_numpy()
y = train['Survived'].to_numpy()

X = X.reshape((X.shape[0], X.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(X, y)



model = keras.Sequential(
    [

        layers.Conv1D(filters= 8, kernel_size= 3, activation="relu", input_shape= (X.shape[1],1)), # 11 -3 +1 = 9

        layers.Conv1D(filters= 16, kernel_size= 4, activation="relu"), # 9 -4 +1 = 6

        layers.MaxPooling1D(2), # 6/2 = 3

        layers.GlobalMaxPooling1D(), # 1

        layers.Flatten(),

        layers.Dropout(0.25),

        layers.Dense(1, activation='sigmoid'),
        
    ])




model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
