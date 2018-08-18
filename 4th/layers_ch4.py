# Chap3で使うNN関連のプログラム(layers)
import numpy as np
import matplotlib.pyplot as plt
from functions_ch4 import UnigramSampler, cross_entropy_error
import time
import sys

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

class SigmoidWithLoss:
    '''
    二値分類問題に使用される損失関数付きのシグモイド関数
    入力はバッチ×1次元（2次元配列ででてくるのか分からん）
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        print('shape = {0}'.format(self.y.shape))
        print('np.c_[1 - self.y, self.y] = {0}'.format(np.c_[1 - self.y, self.y]))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t) 
        # np._cの意味は，2次元以上の配列で、最も最低の次元(axisの番号が一番大きい)の方向で配列を結合する
        # ただ，特殊で，1次元の時は列に結合される
        # おそらくこの場合はTが，バッチサイズのみになっているので，それに合わせているんだと思う，つまりは列結合

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0] # しつこいけど行数はバッチ数

        # バッチ×1
        dx = (self.y - self.t) * dout / batch_size # バッチサイズで割っておく（最終的には重みの勾配計算するときに全部足し算されるから）

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
        dw[...] = 0 # そのまま値をリセット

        print('embeding shape = {0}'.format(dw))

        for i, word_id in enumerate(self.idx): # idを取り出す
            dw[word_id] += dout[i] 
        # 加算なのはrepeatノードとして考えてもそうですが，Matmulの一部の動作を取り出しているので，足し算しないと話がおかしくなります
        # matmulを実際に同じ要素を含む形で書いてみると加算理由がわかるかと思います
        # ちなみにこれNoneなのはこれ以上逆伝播する必要がないからです
        return None

    
class EmbeddingDot:
    '''
    出力層のMatmulの代替え
    基本はembedinglayerでそこに少し要素が＋された感じ
    '''
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None # forward計算結果を保存

    def forward(self, h, idx):
        target_W = self.embed.forward(idx) # 取り出した重み
        out = np.sum(target_W * h, axis=1)# バッチ対応です（axis = 1）で各バッチに対応する内積が出ます

        self.cache = (h, target_W) # どれを取り出すのか保存しておく, tupleなのは変更されないように？

        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1) # バッチ×列にしている（sigmoidlossから戻ってくるものに補正）
        
        dtarget_W = dout * h # 内積の逆伝播は内積です

        print('h = {0}'.format(h))
        print('dout = {0}'.format(dout))
        print('dtarget_W = {0}'.format(dtarget_W))

        self.embed.backward(dtarget_W) # ある特定の場所だけの更新がかかる用の行列ができたので，それをembeddingに渡せばいい

        dh = dout * target_W # 本流用，結局この要素しかいらないからこの計算式で更新

        return dh

class NegativeSamplingLoss():
    '''
    Negative sampling 付きの最終レイヤーを作成

    '''
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size # 何個negativesamplingを行うのか
        self.sampler = UnigramSampler(corpus, power, sample_size) # ネガティブサンプリング器作成
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)] # 省略できるんだ．．iとか知らなかった，レイヤー作成
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)] # レイヤー作成　+1してるのは正解の分

        self.params, self.grads = [], [] # ここはいつも通りに収納するためのリスト
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0] # 行がバッチ数，targetはidです
        negative_sample = self.sampler.get_negative_sample(target) # ターゲットを除いたうえでサンプルを作成する

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32) #正解ラベルなので1
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例のフォワード
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i] #誤りラベルなので0，サンプリング数分作成する
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout) # 重りの更新は別で行っているように見えるが，作成時に同じものをwとしていれているので，Wは1つ
            dh += l1.backward(dscore) # 結局分岐ノードなので最後は足し算

        return dh
