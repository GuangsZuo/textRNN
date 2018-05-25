import numpy as np
import pandas as pd
def load_word_embeding(embeding_file):
    def get_coef(word, *coefs):
        return word, np.asarray(coefs, dtype=np.float32)
    return dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file))

def tokenize(corpus, max_words, max_len):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(corpus)
    words_index = tokenizer.texts_to_sequences(corpus)
    words_index = pad_sequences(words_index, maxlen=max_len)
    return words_index

def get_word_embding(word_index, embeding_file, max_words, embed_size, unk="something"):
    max_words = min(max_words, len(word_index)) + 1
    embeding_matrix = np.zeros((max_words, embed_size))
    embeding_dict = load_word_embeding(embeding_file)
    lose = 0
    for word, i in word_index.items():
        if word not in embeding_dict:
            lose += 1
            word = unk
        if i > max_words:
            continue
        embeding_matrix[i] = embeding_dict[word]
    return embeding_matrix, max_words

if __name__ == "__main__":
    neg = pd.read_csv("./MR/mr.neg", delimiter="\n", encoding="ISO-8859-1")
    pos = pd.read_csv("./MR/mr.pos", delimiter="\n", encoding="ISO-8859-1")



