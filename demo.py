from text_processing import *
import os
embeding_file="./glove.840B.300d.txt"
embed_size = 300
max_words = 20000
sentence_size = 30

model_file = "./best_model.hdf5"

is_first = 1

if is_first:
    neg = pd.read_csv("./data/MR/mr.neg", delimiter="\n", encoding="ISO-8859-1", header=None)
    pos = pd.read_csv("./data/MR/mr.pos", delimiter="\n", encoding="ISO-8859-1", header=None)

    neg = neg.rename(index=str, columns={0: "text"})
    pos = pos.rename(index=str, columns={0: "text"})

    neg["label"] = 0
    pos["label"] = 1

    data = pd.concat([neg,pos]).sample(frac=1).reset_index(drop=True)
    labels = data["label"].reshape(-1,1)
    data, word_index = tokenize(data["text"], max_words, sentence_size)
    train_data, test_data = data[:data.shape[0]-100], data[-100:]

    word_embeding,max_words = get_word_embding(word_index, embeding_file, max_words, embed_size)

    np.save("embeding-300d",word_embeding)
    np.save("train_data",train_data)
    np.save("test_data",test_data)
    np.save("labels",labels)
else:
    word_embeding = np.load("embeding-300d.npy")
    train_data = np.load("train_data.npy")
    test_data = np.load("test_data.npy")
    labels = np.load("labels.npy")
    max_words = word_embeding.shape[0]

from algo import textrnn_algo
import tensorflow as tf
algo = textrnn_algo(train_data,labels[:train_data.shape[0]], word_embeding, model_file, classes=1, sentence_length=sentence_size,
                    embed_size=embed_size, target_is_prob=False, word_size=max_words,
                    is_sigmoid_loss=True, learning_rate=1e-3, epoch=100, early_stopping_rounds=3,
                    batch_size=64)
algo.model_train()
result = algo.model_predict(test_data)
print(result)