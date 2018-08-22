# Chap3で使うNN関連のプログラム(ネットワーク)
import numpy as np
import matplotlib.pyplot as plt
from functions_ch5 import UnigramSampler, cross_entropy_error
from layers_ch5 import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss, TimeRNN
import time
import sys

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        # 基本はこの式
        # x（バッチ×時系列×次元） --> x * Wx(Embedding) -->  hWh + xWx + b = h --> h（バッチ×時系列×次元）* Wx(Affine) --> 出力
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1] # 単純に配列の順番

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers: # 今までと同様の手法でぶち込む
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
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


class Trainer:
    '''
    いろいろまとめっちゃった学習クラス
    '''
    def __init__(self, model, optimizer): 
        self.model = model # 上記のクラスが入るイメージ
        self.optimizer = optimizer # 今回はadam
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch, batch_size, max_grad=None, eval_interval=20): # デフォルト設定いるのか疑問,消した
        data_size = len(x) # 行×列の行がでるイメージ
        max_iters = data_size // batch_size # 切り捨て除算らしい
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer 
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            # 単純に混ぜただけ

            for iters in range(max_iters):
                # 順番に取り出していく
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                # print(loss)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約，下参照
                if max_grad is not None:# RNNで使用
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads) # 片方だけ更新すれば全部更新されます(共有された重みはアドレスを共有しているので)
                total_loss += loss
                loss_count += 1

                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count # そのエポックでのロスの平均
                    elapsed_time = time.time() - start_time # 計算時間の計算
                    print('epoch {0}, iter {1} / {2} , time {3}[s] , loss {4}'
                          .format(self.current_epoch + 1, iters + 1, max_iters, round(elapsed_time, 3), round(avg_loss, 3) ))
                    self.loss_list.append(float(avg_loss)) # epoch毎のロス
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()

class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # パープレキシティの評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()


def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    加算するのは，今回でいえば共有するものが2つあるからって感じ
    誤差自体というか傾きは重みを共有している分だけ変化することになる（共有するってことはそういうこと），それぞれで更新されるので
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params) #

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]: # 何番目のレイヤー同士が一緒かをみてる・ここで重みが同じか判定(is 演算子はオブジェクトが同一か判定します)  == は値
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j) # 取り除く（レイヤーの集合から）
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break # popするので，これでよい（各要素の最初の部分だけみることになるけど）
            if find_flg: break

        if not find_flg: break # 共通部分がない場合

    return params, grads

def clip_grads(grads, max_norm):
    '''
    勾配クリッピング
    RNNで使います
    ある勾配より大きくなったらもう使わないってやつ
    '''
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


# optimizer
class Adam:
    '''
    パラメータを更新する手法の1つ
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        # たぶんだけど，イタレーション回数が増えるにつれて，更新量が小さくなるようにしているはず(下の方のmainで確認済み)
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)): # 各要素ごとにみる
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


if __name__ == '__main__':
    # 本当に，更新のたびにadamsのalphaが小さくなってるか確認
    iterations = 100

    lr=0.001
    beta1=0.9
    beta2=0.999
    lr_traj = []
    
    for i in range(1, iterations):
        # たぶんだけど，イタレーション回数が増えるにつれて，更新量が小さくなるようにしているはず
        lr = lr * np.sqrt(1.0 - beta2**i) / (1.0 - beta1**i)

        lr_traj.append(lr)
        
    plt.plot(range(1, iterations), lr_traj)
    plt.show()

