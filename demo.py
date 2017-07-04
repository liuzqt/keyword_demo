# encoding: utf-8

'''

@author: ZiqiLiu


@file: demo.py

@time: 2017/6/14 下午5:04

@desc:
'''
import os
import asyncio

import time
import tornado
import tornado.web
from tornado.platform.asyncio import AsyncIOMainLoop
from positional_encoding import positional_encoding_op
import numpy as np
import tensorflow as tf
# from config.config import get_config
from config.ctc_config import get_config
from utils.common import path_join
from utils.prediction import moving_average, decode, predict, ctc_predict
from fetch_wave import fetch
from normalize import main
import librosa

# load graph
config = get_config()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def process_wave(f):
    y, sr = librosa.load(f, sr=config.samplerate)

    mel_spectrogram = np.transpose(
        librosa.feature.melspectrogram(y, sr=sr, n_fft=config.fft_size,
                                       hop_length=config.step_size,
                                       power=config.power,
                                       fmin=300,
                                       fmax=8000,
                                       n_mels=60))

    return mel_spectrogram, y


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
        # with self.sess as sess, self.graph.as_default():
        prob = self.sess.run(['model/softmax:0'],
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

    def predict_ctc(self, inputX):
        seqLen = np.asarray([len(inputX)])
        # with self.sess as sess, self.graph.as_default():
        output = self.sess.run(['model/dense_output:0'],
                               feed_dict={'model/inputX:0': inputX,
                                          'model/seqLength:0': seqLen})
        np.set_printoptions(precision=4, threshold=np.inf,
                            suppress=True)
        # print(prob)
        result = ctc_predict(output[0])
        return True if result == 1 else False


def run(device_id='32EFEA3263D079E1BE3767C87FC0A1C2', current=False):
    config = get_config()

    runner = Runner(config)
    label = ""
    if not current:
        wave, label = fetch(device_id)
        print('wave', wave)
    spec, _ = process_wave('temp.wav')
    result = runner.predict(spec)

    print(result, label)


class HotWordHandler(tornado.web.RequestHandler):
    def initialize(self, runner):
        self.runner = runner

    def get(self):
        device_id = self.get_argument('device_id')
        wave, label, wave_id = fetch(device_id)
        print('wave', wave)
        main()
        spec, _ = process_wave('normalized-temp.wav')
        result = self.runner.predict_ctc(spec)
        self.write({
            'result': result,
            'label': label,
            'time': time.strftime('%Y-%m-%d %A %X %Z',
                                  time.localtime(time.time())),
            'wave_id': wave_id
        })


def start_server():
    runner = Runner(config)
    AsyncIOMainLoop().install()
    app = tornado.web.Application([
        (r'/()', tornado.web.StaticFileHandler, {
            'path': BASE_DIR,
            'default_filename': 'index.html'
        }),
        (r'/api/hotword', HotWordHandler, {'runner': runner}),
    ], debug=True
    )
    app.listen(8080)
    print('start server')
    loop = asyncio.get_event_loop()
    loop.run_forever()


if __name__ == '__main__':
    start_server()
    # run('32EFEA3263D079E1BE3767C87FC0A1C2', True)
