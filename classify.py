import pandas
import keras

import word2vec

MAX_SEQ_LENGTH = 100

word2vec_model = word2vec.run()

data_df = pandas.read_csv("data/train.csv")

print("Loaded data into a dataframe")


def create_embedding_layer():
    vocab_size = len(word2vec_model.wv.vocab)
    output_dim = len(word2vec_model[word2vec_model.wv.vocab[0]])
    embedding_layer = keras.layers.Embedding(
        vocab_size, output_dim, input_length=MAX_SEQ_LENGTH
    )

    return embedding_layer

def tokenize_data():
    train_size = int(len(data_df) * .8)

    # Use numpy.concatenate to build feature vector

    train_X = data_df[['question1', 'question2']][:train_size]
    train_Y = data_df['is_duplicate'][:train_size]

    val_X = data_df[['question1', 'question2']][train_size:]
    val_Y = data_df['is_duplicate'][train_size:]
