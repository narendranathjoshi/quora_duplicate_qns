import pandas
import keras

data_df = pandas.read_csv("data/train.csv")

print("Loaded data into a DF")

train_size = int(len(data_df) * .8)
 
train_X = data_df[['question1', 'question2']][:train_size]
train_Y = data_df['is_duplicate'][:train_size]
 
val_X = data_df[['question1', 'question2']][train_size:]
val_Y = data_df['is_duplicate'][train_size:]

