# layers
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

class RNN():
    '''
    RNNの1ステップの処理を行うレイヤーの実装
    '''
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b] # くくっているのは同じ理由
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev): # h_prevは1つ前の状態
        Wx, Wh, b = self.params # 取り出す
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b

        h_next = np.tanh(t) # tanh関数

        self.cache = (x, h_prev, h_next) # 値を保存しておく
        
        return h_next
    
    def backward(self, dh_next): # 隠れ層の逆伝播の値が引数
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2) # tanhの逆伝播（各要素に対してかかる）
        db = np.sum(dt, axis=0) # いつものMatmulと同じ
        dWh = np.dot(h_prev.T, dt) # いつものMatmulと同じ
        dh_prev = np.dot(dt, Wh.T) # 上の式みて考えれば分かる
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx # 値をコピー
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    '''
    上のやつ全部まとめたやつBPTTさせる分
    '''
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b] # くくっているのは同じ理由 hWh + xWx + b = h
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params # パラメータの初期化
        N, T, D = xs.shape # xsの形, Dは入力ベクトルの大きさ，このレイヤーはまとめてデータをもらうので！

        D, H = Wx.shape 

        self.layers = [] # 各レイヤー（RNNの中の）
        hs = np.empty((N, T, H), dtype='f') # Nはバッチ数，Tは時間数，HがHの次元

        if not self.stateful or self.h is None: # statefulでなかったら,または，初期呼び出し時にhがなかったら（前の状態を保持しなかったら）
            self.h = np.zeros((N, H), dtype='f') # Nはバッチ数

        for t in range(T): # 時間分（backpropする分）だけ繰り返し
            layer = RNN(*self.params) # 可変長引数らしい　ばらばらで渡される今回のケースでいえば，Wx, Wh, bとしても同義
            self.h = layer.forward(xs[:, t, :], self.h) # その時刻のxを渡す
            hs[:, t, :] = self.h # 保存しておく
            self.layers.append(layer) # RNNの各状態の保存

        return hs

    def backward(self, dhs): 
        Wx, Wh, b = self.params # パラメータの初期化
        N, T, H = dhs.shape # xsの形, Dは入力ベクトルの大きさ，このレイヤーはまとめてデータをもらうので！
        D, H = Wx.shape 

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t] # 一つずつ保存しておいたlayerを呼び出す
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 勾配は合算します（分岐ノードなので）
            dxs[:, t ,:] = dx

            for i, grad in enumerate(layer.grads): # 各重み(3つ，Wx, Wb, b)を取り出す，同じ重みを使っているので，勾配はすべて足し算
                grads[i] += grad 
        
        # print(len(grads))

        for i, grad in enumerate(grads): # 時系列順に並んでいるやつをコピー
            self.grads[i][...] = grad # 
        
        self.dh = dh

        return dxs # 後ろに逆伝播させる用(N, T, D)になっている

class TimeAffine:
    '''
    AffineがT個分ある（行列演算レベルでくっつけてある）
    '''
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1) # 2次元に変換⇒次元だけは守っているイメージ，バッチっていう概念がなくなる感じ
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1) # 時系列データが出力される

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout) # こうすれば，横向きになっているから全部勾配が勝手に足される（forで回す必要がない）行×列でいける(D * N*T) * (N*H * H)かな
        dx = np.dot(dout, W.T) # こっちもおなじ原理
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class TimeIdentifyWithLoss:
    '''
    時系列データをまとめて受け付ける損失関数
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.counter = 0

    def forward(self, xs, ts):
        N, T, D = xs.shape # ここでDは1

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, D) # ここは同じになるはず
        ts = ts.reshape(N * T, D) #

        ys = copy.deepcopy(xs) # 恒等関数
        
        loss = 0.5 * np.sum((ys - ts)**2)
        loss /= N * T # 1データ分での誤差

        # print('Y = {0}, T = {1}'.format(np.round(ys, 3), np.round(ts, 3)))
        # print('N * T = {0}'.format(N*T))
        # print('loss = {0}'.format(loss))
        # if self.counter % 1 == 0:
            # plt.plot(range(len(ys.flatten())) , ys.flatten())
            # plt.plot(range(len(ys.flatten())) , ts.flatten())
            # plt.show()
        # a = input()

        self.cache = (ts, ys, (N, T, D))
        self.counter += 1
        return loss

    def backward(self, dout=1):
        ts, ys, (N, T, D) = self.cache

        dx = ys - ts # 出力をこっちにいれとく
        dx /= N * T

        dx = dx.reshape((N, T, D))

        return dx