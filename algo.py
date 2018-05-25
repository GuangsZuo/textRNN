import tensorflow as tf
import numpy as np

from model import textrnn

class textrnn_algo:
    def __init__(self, train_data, word_embeding, model_file,
                 classes, sentence_length, embed_size, target_is_prob, word_size,
                 is_sigmoid_loss=True, learning_rate=1e-3, optimizer=tf.train.AdamOptimizer, **kwargs):
        self.data = train_data
        self.model = textrnn(classes, sentence_length, embed_size, target_is_prob, word_size,
                             is_sigmoid_loss, learning_rate, optimizer, word_embeding)
        self.sess = tf.Session()
        self.saver= tf.train.Saver()
        self.model_file = model_file
        self.classes = classes
        if "epoch" not in kwargs:
            self.epoch = 100
        else:
            self.epoch = int(kwargs["epoch"])
        if "early_stopping_rounds" not in kwargs:
            self.early_stopping_rounds = 3
        else:
            self.early_stopping_rounds = int(kwargs["epoch"])
        if "batch_size" not in kwargs:
            self.batch_size = 64
        else:
            self.batch_size = int(kwargs["batch_size"])

        self.writer = tf.summary.FileWriter("./tflogs/", self.sess.graph)

    def model_train(self):
        x_train, y_train, x_val, y_val = self.split_train_data(self.data)
        for epoch in range(1, self.epoch+1):
            best_score = 2147483644
            bad_rounds = 0
            train_loss = 0
            for st,ed in zip(range(0,x_train.shape[0], self.batch_size), range(self.batch_size, x_train.shape[0]+self.batch_size, self.batch_size)):
                x_train_batch = x_train[st:ed]
                y_train_batch = y_train[st:ed]
                losses, _ = self.sess.run([self.model.losses, self.model.trainop],
                              feed_dict={self.model.input: x_train_batch, self.model.target: y_train_batch})
                train_loss += np.sum(losses)
            train_loss = train_loss / x_train.shape[0] / self.classes
            val_loss = self.evaluate_model(x_val, y_val)
            print("Epoch %d: train loss : %.6f, val loss: %.6f" % (epoch, train_loss, val_loss))
            if val_loss < best_score:
                print("*** New best score ***\n")
                best_score = val_loss
                bad_rounds = 0
                self.saver.save(self.sess, self.model_file)
            else:
                bad_rounds += 1
                if bad_rounds >= self.early_stopping_rounds:
                    print("Epoch %05d: early stopping, best score = %.6f" % (epoch, best_score))
                    break

    def evaluate_model(self, x_val, y_val, batch_size=1024):
        val_loss = 0
        for st, ed in zip(range(0, x_val.shape[0], batch_size),
                          range(batch_size, x_val.shape[0] + batch_size, batch_size)):
            x_val_batch = x_val[st:ed]
            y_val_batch = y_val[st:ed]
            losses, _ = self.sess.run([self.model.losses, self.model.trainop],
                                      feed_dict={self.model.input: x_val_batch, self.model.target: y_val_batch})
            val_loss += np.sum(losses)
        val_loss = val_loss / x_val.shape[0] / self.classes
        return val_loss

    def split_train_data(self, data):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(nsplits=10)
        train_index, val_index = skf.split(data["text"], data["label"])[0]
        return data["text"][train_index], data["label"][train_index], data["text"][val_index], data["label"][val_index]

    def model_predict(self, test_data, batch_size=1024):
        self.saver.restore(self.sess, self.model_file)
        result = np.zeros((test_data.shape[0],self.classes))
        for st, ed in zip(range(0, test_data.shape[0], batch_size),
                          range(batch_size, test_data.shape[0] + batch_size, batch_size)):
            x_val_batch = test_data["text"][st:ed]
            output = self.sess.run([self.model.output], feed_dict={self.model.input: x_val_batch})
            result[st:ed] = output
        return result









