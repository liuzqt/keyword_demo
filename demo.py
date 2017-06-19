# encoding: utf-8

'''

@author: ZiqiLiu


@file: demo.py

@time: 2017/6/14 下午5:04

@desc:
'''
import tensorflow as tf
from config.config import get_config
from utils.common import path_join
from utils.prediction import moving_average, decode, predict
from process_wav import process_wave
import numpy as np
from fetch_wave import fetch
from io import  BytesIO

# load graph
class Runner():
    def __init__(self, config):
        self.graph_def = tf.GraphDef()
        self.config = config
        with open(path_join(config.graph_path, config.graph_name), 'rb') as f:
            # print (f.read())
            self.graph_def.ParseFromString(f.read())
        # for node in self.graph_def.node:
        #     print(node.name)

        self.sess = tf.Session()
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")
        self.sess = tf.Session(graph=self.graph)

    def predict(self, inputX):
        seqLen = np.asarray([len(inputX)])
        with self.sess as sess, self.graph.as_default():
            prob = sess.run(['model/softmax:0'],
                            feed_dict={'model/inputX:0': inputX,
                                       'model/seqLength:0': seqLen})
            np.set_printoptions(precision=4, threshold=np.inf,
                                suppress=True)
            # print(prob)
            with open('logits.txt', 'w') as f:
                f.write(str(prob))
            moving_avg = moving_average(prob[0], self.config.smoothing_window,
                                        padding=True)

            prediction = predict(moving_avg, self.config.trigger_threshold,
                                 self.config.lockout)
            result = decode(prediction, self.config.word_interval,
                            self.config.golden)
        return True if result == 1 else False


def run(device_id='32EFEA3263D079E1BE3767C87FC0A1C2',current=False):
    config = get_config()

    runner = Runner(config)
    label=""
    if not current:
        wave, label = fetch(device_id)
        print('wave', wave)
    spec, _ = process_wave('temp.wav')
    result = runner.predict(spec)

    print(result, label)

if __name__ == '__main__':
    run('32EFEA3263D079E1BE3767C87FC0A1C2')
