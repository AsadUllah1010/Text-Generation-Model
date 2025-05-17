import tensorflow as tf
import numpy as np

# Download the text file
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Read the file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f"Length of text: {len(text)} characters")

# Get unique characters
vocab = sorted(set(text))
vocab_size = len(vocab)

# Create char<->int mappings
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Encode the text as integers
text_as_int = np.array([char2idx[c] for c in text])

# Sequence length for training
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create training dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Create sequences of seq_length + 1
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Function to split into input and target
def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq

dataset = sequences.map(split_input_target)

# Shuffle and batch
BATCH_SIZE = 64
BUFFER_SIZE = 10000
embedding_dim = 256       
rnn_units = 1024  

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.Input(batch_shape=(batch_size, None)),  # define input shape here
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)

model.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer='adam', loss=loss)

# Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = checkpoint_dir + "/ckpt_{epoch}.weights.h5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Train the model
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

import os
import glob

# Rebuild model for generation with batch size = 1
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Load the latest .weights.h5 file
latest = max(glob.glob(checkpoint_dir + "/*.weights.h5"), key=os.path.getctime)
model.load_weights(latest)

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string, num_generate=500):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0  # higher: more random; lower: more predictable

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

# Generate a sample
print(generate_text(model, start_string="ROMEO: "))

