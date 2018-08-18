# Chap3で使うNN関連のプログラム(ネットワーク)
import numpy as np
import matplotlib.pyplot as plt
from functions_ch4 import UnigramSampler, cross_entropy_error
from layers_ch4 import NegativeSamplingLoss, Embedding
import time
import sys


class CBOW:
    '''
    ネットワークの作成
    '''
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype('f') # 正規分布
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # レイヤ作成
        self.in_layers = []
        for i in range(2 * window_size): # コンテキストの数に応じて（windowsizeが1なら両サイドあることになるので2倍）
            layer = Embedding(W_in) # 同じ重みを参照する
            self.in_layers.append(layer)

        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss] # これで要素が足し算される
        
        # 各レイヤーの重みをまとめておく
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 取り出すとき用
        self.word_vecs = W_in

    def forward(self, contexts, target): # 前章まではここはone-hotだったが，その計算を工夫してなくしたのでone-hotにする必要はない
        h = 0
        for i, layer in enumerate(self.in_layers): 
            h += layer.forward(contexts[:,i]) # 何列目を取り出すか（しつこいですが行数がバッチ）
        
        h *= 1 / len(self.in_layers) # 前回で0.5かけたのと同じで，各レイヤーからの出力を等倍で考慮，hiddensizeで来る
        loss = self.ns_loss.forward(h, target) # targetはようは答えです

    def backward(self, dout=1): # 同様にスタートは1
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers) # この作業も同じ（逆伝播させているので，かけた分かけます（掛け算のノードと同じ））
        
        for layer in self.in_layers: # 後は順番に呼び出すだけ
            layer.backward(dout)

        return None


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

    def fit(self, x, t, max_epoch, batch_size, max_grad=None, eval_interval=20): # デフォルト設定いるのか疑問消した
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

