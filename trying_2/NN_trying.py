import sys
import numpy as np
from layers_trying import TimeRNN, TimeAffine, TimeIdentifyWithLoss

class SimpleRnn:
    '''
    今回使用するネットワーク
    '''
    def __init__(self, input_size, hidden_size, output_size):
        D, H, O = input_size, hidden_size, output_size # 入力の次元，隠れ層の次元，出力の次元
        rn = np.random.randn

        # 重みの初期化
        rnn_Wx = (rn(D, H) / 10).astype('f')
        rnn_Wh = (rn(H, H) / 10).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, O) / 10).astype('f')
        affine_b = np.zeros(O).astype('f')

        # レイヤの生成
        self.layers = [
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True), 
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeIdentifyWithLoss()
        self.rnn_layer = self.layers[0]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts): # 教師，入力ともに三次元
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()
