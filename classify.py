# libraries
import pandas
import keras
import gensim
import numpy as np

# local modules
import word2vec

def create_embedding_layer():
    vocab_size = len(word2vec_model.wv.vocab)
    output_dim = len(word2vec_model[word2vec_model.wv.vocab[0]])
    embedding_layer = keras.layers.Embedding(
        vocab_size, output_dim, input_length=MAX_SEQ_LENGTH
    )

    return embedding_layer

def tokenize_data(data_df):
    def vectorize_questions(row):
        tokens = list(gensim.utils.tokenize(row['question1'], lowercase=True))
        tokens2 = list(gensim.utils.tokenize(row['question2'], lowercase=True))
        
        if tokens and tokens2:
            vectors = np.array(word2vec_model[tokens[0]])
                    
            for t in tokens[1:]:
                vectors = np.concatenate((vectors, word2vec_model[t]), axis=None)

            for t2 in tokens2:
                vectors = np.concatenate((vectors, word2vec_model[t2]), axis=None)
            
            return vectors

        else:
            return []



    print("Tokenizing data ...")

    data_df['vector'] = data_df.apply(vectorize_questions, axis=1)

    print(data_df['vector'][0])

    X_y = data_df.values(columns=['vector'])

    print(X_y.shape)

    # train_size = int(len(data_df) * .8)

    # train_X = data_df[['question1', 'question2']][:train_size]
    # train_Y = data_df['is_duplicate'][:train_size]

    # val_X = data_df[['question1', 'question2']][train_size:]
    # val_Y = data_df['is_duplicate'][train_size:]

if __name__ == "__main__":
    MAX_SEQ_LENGTH = 100
    word2vec_model = word2vec.run()
    data_df = pandas.read_csv("data/train.csv")
    print("Loaded data into a dataframe")
    
    tokenize_data(data_df)
