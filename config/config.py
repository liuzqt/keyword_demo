# encoding: utf-8

'''

@author: ZiqiLiu


@file: config.py

@time: 2017/5/18 上午11:18

@desc:
'''
import argparse


def get_config():
    return Config()


class Config(object):
    def __init__(self):
        # basic options
        self.mode = "train"  # train,valid
        self.ktq = False
        self.spectrogram = 'mel'  # mfcc,mel
        self.label_id = 0  # nihaolele,lele,whole
        self.label_list = ['nihaolele', 'lele', 'whole']
        self._num_classes = [3, 2, 2]  # word+1 for background
        self._golden = [[2, 1], [1], [1]]
        self.reset_global = 0
        self.gpu = "0"

        # path flags

        self.model_path = './params/mel_stft/'
        self.save_path = './params/mel_stft/'
        self.graph_path = './graph/mel/'
        self.graph_name = 'graph.pb'
        self.data_path = './data/mel_stft/'

        self.data_path = '/ssd/liuziqi/mel_all_stft/'
        self.model_name = 'best5.ckpt'
        self.rawdata_path = './rawdata/'
        self.rawdata_path = '/ssd/keyword/'
        # self.data_path = './test/data/azure_garbage/'


        # pre-processing flags
        self.fft_size = 400
        self.step_size = 160
        self.samplerate = 16000
        self.max_sequence_length = 2000
        self.power = 1
        self.fmin = 300
        self.fmax = 8000

        # noise flags
        self.use_white_noise = False
        self.use_bg_noise = False
        self.bg_noise_prob_raise = 1.05
        self.bg_decay_max_db = -6
        self.bg_decay_min_db = -20
        self.bg_noise_prob = 0.5

        # model params
        self.cell_clip = 3.
        self.num_layers = 2
        self.learning_rate = 5e-3
        self.max_grad_norm = -1
        self.num_features = 60
        self.hidden_size = 64
        self.use_project = False
        self.num_proj = 32
        self.max_epoch = 200
        self.drop_out_input = -1
        self.drop_out_output = -1
        self.lr_decay = 0.8
        self.decay_step = 10000
        self.use_relu = False

        self.optimizer = 'adam'  # adam sgd nesterov
        self.max_pooling_loss = False
        self.max_pooling_standardize = True

        # training flags
        self.batch_size = 32
        self.tfrecord_size = 32
        self.valid_steps = 320


        # these three sizes are frames, which depend on STFT frame size
        self.trigger_threshold = 0.5  # between (0,1), but this param is somehow arbitrary
        self.smoothing_window = 9
        self.latency = 30
        self.word_interval = 70
        self.lockout = 50

    @property
    def label(self):
        return self.label_list[self.label_id]

    @property
    def num_classes(self):
        # word+1 for background
        return self._num_classes[self.label_id]

    @property
    def golden(self):
        return self._golden[self.label_id]

    def show(self):
        for item in self.__dict__:
            print(item + " : " + str(self.__dict__[item]))
