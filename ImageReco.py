import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, Y_train),(X_test,Y_test) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(X_train[0])
#plt.show()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

md = tf.keras.models.Sequential()
md.add(tf.keras.layers.Flatten())

md.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
md.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
md.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

md.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

md.fit(X_train, Y_train, epochs=3 )

val_loss, val_acc = md.evaluate(X_test,Y_test)

print(val_loss)
print(val_acc)