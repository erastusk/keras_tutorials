import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



#***************************************************************
#		IMPORT DATA
#***************************************************************

imdb = keras.datasets.imdb

'''
Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed,
 and each review is encoded as a sequence of word indexes (integers). For convenience, words are indexed by overall 
 frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. 
 This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate 
 the top 20 most common words".
num_words = 10,000 means the top 10k words in the corpus.
Top 10k most frequent words.
'''

#***************************************************************
#		TRAIN TEST DATA SPLIT
#***************************************************************

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

'''
imdb.load does the following for you:

*   Removes punctuations
*   Lemmetizes words
*   Also, seems to maintain contractions, eg won't, didn't 

But still maintains the sentence as it were.
'''

# Inspect your data
# print(x_train[0], y_train[0])

# Print largest sentence word count

len_sent = [len(i) for i in x_train]
print(len_sent[np.argmax(len_sent)])

'''
Reverse index to actual word function: below
* 1 - The
* 6 - Is
* 964 - rate
* etc
'''
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
	return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(x_train[15])

print(decode_review(x_train[15]))

'''
keras.preprocessing.sequence.pad_sequences(sequences, 
											maxlen=None, 
											dtype='int32', 
											padding='pre', 
											truncating='pre', 
											value=0.0)

*   sequences: List of lists, where each element is a sequence.
*   maxlen: Int, maximum length of all sequences.
*   dtype: Type of the output sequences. To pad sequences with variable length strings, you can use object.
*   padding: String, 'pre' or 'post': pad either before or after each sequence.
*   truncating: String, 'pre' or 'post': remove values from sequences larger than maxlen, either at the beginning or 
	at the end of the sequences.
*   value: Float or String, padding value.

'''
#***************************************************************
#		DATA PREPARATION FOR MODEL
#***************************************************************

x_train = keras.preprocessing.sequence.pad_sequences(x_train,
													 maxlen=256,
													 padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test,
													maxlen=256,
													padding='post')

'''
* If dealing with a text classification problem, prefer to start with an Embedding layer followed by a Global 
	average layer, then feed them to your Dense layers.
* You don't need to provide input shape with embedded layers, just the size of the vocabulary = largest int in the 
	freqdist, in our case its 10000.
* It's however assumed the features are of the same length, hence the need for padding. Now when the dense layers come 
	into play, they already know the shape.
* If you you were going directly to a dense layer an input shape would be required, in our case (256,)

'''
#***************************************************************
#		MODEL - DEFINE MODEL ARCHITECTURE
#***************************************************************

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

'''
# Embedding

* input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1. For example if you have an array, what's
 	the maximum integer, below should be 1k, range of ints created from.
* output_dim: int >= 0. Dimension of the dense embedding.
* embeddings_initializer: Initializer for the embeddings matrix (see initializers).
* embeddings_regularizer: Regularizer function applied to the embeddings matrix (see regularizer).
* activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
* embeddings_constraint: Constraint function applied to the embeddings matrix (see constraints).
* mask_zero: Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful
 	when using recurrent layers which may take variable length input. If this is True then all subsequent layers in the 
 	model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, 
 	index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).
* input_length: Length of input sequences, when it is constant. This argument is required if you are going to connect
 	Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).


model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
* the model will take as input an integer matrix of size (batch, input_length).
* the largest integer (i.e. word index) in the input should be
* no larger than 999 (vocabulary size).
* now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)



# Dense

Just your regular densely-connected NN layer.

* as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
* now the model will take as input arrays of shape (*, 16)
* and output arrays of shape (*, 32)
* after the first layer, you don't need to specify
* the size of the input anymore:
model.add(Dense(32))

# Example
* as first layer in a sequential model:

```
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
```


* now the model will take as input arrays of shape (*, 16)
* and output arrays of shape (*, 32)
* after the first layer, you don't need to specify
* the size of the input anymore:

model.add(Dense(32))
'''

#***************************************************************
#		MODEL - COMPILATION ( OPTIMIZER, LOSS)
#***************************************************************

model.compile(optimizer=tf.optimizers.Adam(),
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

'''
* binary_crossentropy being used because we only have two classess.
* sparse_categorical_crossentropy with multiple classes
'''

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Further divide training set by extracting validation tests for use after training

#***************************************************************
#		MODEL - FIT
#***************************************************************

history = model.fit(partial_x_train,
					partial_y_train,
					epochs=40,
					batch_size=512,
					validation_data=(x_val, y_val),
					verbose=1)


#***************************************************************
#		MODEL - FIT
#***************************************************************

results = model.evaluate(x_test, y_test)
#print(results)

history_dict = history.history
history_dict.keys()

# Map val_loss vs Epoch to see where loss starts to tail off, tells you where to stop fitting (# of epochs)

plt.clf()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.plot(epochs, acc, 'go', label='Training Accuracy')
plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
plt.title('Training/Validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#***************************************************************
#		MODEL - PREDICT
#***************************************************************

predictions = model.predict(x_test)
print(predictions)