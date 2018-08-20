# Chap3で使うNN関連のプログラム(layers)
import numpy as np
import matplotlib.pyplot as plt
from functions_ch5 import UnigramSampler, cross_entropy_error, softmax
import time
import sys

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
    上のやつ全部まとめたやつTime分
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

        # print(N, T, D)
        # sys.exit()

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

class MatMul:
    def __init__(self, W):
        self.params = [W] # わざわざこうしてるのは，layersで処理するときに配列同士の足し算がうまくいかないから（layer毎の分離が出来なくなる）
        self.grads = [np.zeros_like(W)] # こちも同じ理由　レイヤー毎に分離したいので
        self.x = None

    def forward(self, x):
        # 順伝播
        W, = self.params # こうやると中身取り出せます
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        # 逆伝播
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW # deepコピーと同じ（アドレス固定する）pythonは値に割り振るのでそれを避ける
        return dx

class Embedding:
    '''
    入力層のMatmulの代替え
    '''
    def __init__(self, W):
        self.params = [W] # このくくる理由は前述したとおり
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx # どれを取り出すのか保存しておく
        out = W[idx] # 取り出しただけ
        return out

    def backward(self, dout):
        dw, = self.grads # 取り出し
        dw[...] = 0.0 # そのまま値をリセット

        # print('dw = {0}'.format(dw)) 0になります

        for i, word_id in enumerate(self.idx): # idを取り出す
            dw[word_id] += dout[i] # 取り出したところのを書き換える

        # 加算なのはrepeatノードとして考えてもそうですが，Matmulの一部の動作を取り出しているので，足し算しないと話がおかしくなります
        # matmulを実際に同じ要素を含む形で書いてみると加算理由がわかるかと思います
        # ちなみにこれNoneなのはこれ以上逆伝播する必要がないからです
        return None

    
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape # ここまでは2次元，バッチ×時間
        V, D = self.W.shape # Vは語彙数，これは単語分散行列です

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W) # 同じ重みを共有
            out[:, t, :] = layer.forward(xs[:, t]) # 重み取り出すだけ，形としては列（時間軸がそろっているイメージ），列にそろえて入っていく，これだと左は2次元行列を指しているのでこのままいける
            self.layers.append(layer)

        return out # ここで三次元になる

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0] # layerは重み共有なので足し算

        self.grads[0][...] = grad
        return None

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


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2) # 何番目がもっとも大きいか（[]）の一番小さいところの行でみている

        # これなにやってんだろ
        mask = (ts != self.ignore_label) # -1なら排除してるんだけど，-1のときがあるっぽいな

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T) # indexです，列数
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts]) # 正解のindexだけ取り出している
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -1 * np.sum(ls)
        loss /= mask.sum() # mask部分だけ考える

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys # 出力をこっちにいれとく
        dx[np.arange(N * T), ts] -= 1 # one-hotの場合は，正解のところいがいtは0，しつこいけど行がバッチ！！！！
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx


'''
In [3]: x
Out[3]:
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])

In [4]: x[np.newaxis, :, :] # １つ次元を追加してスライシング。
Out[4]:
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]]])

In [5]: x[:, np.newaxis, :] # axis=1のところに入れることも可能。  
Out[5]:
array([[[ 0,  1,  2,  3,  4]],

       [[ 5,  6,  7,  8,  9]],

       [[10, 11, 12, 13, 14]]])
'''