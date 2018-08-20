# Chap3で使うNN関連のプログラム
import numpy as np
import matplotlib.pyplot as plt
from functions_ch3 import softmax, cross_entropy_error
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

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) # softmaxとかは簡単なので別のファイルで（functions.py）

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t) # （functions.py）
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1 # y-tの式を再現
        dx *= dout
        dx = dx / batch_size # これはbatchsizeで伝播させないと，最終的にいろいろ足し算されてしまうのでそれを防いでいるイメージです（バッチの分が入ってしまう）

        return dx

class SimpleCBOW():
    '''
    シンプルなCBOWのクラス作成
    関数としてmatmulを使うことにする
    '''
    def __init__(self, vocab_size, hidden_size): # ここでの注意点は隠れ層は入力層より小さくしないとだめ
        V, H = vocab_size, hidden_size

        # 重みの初期化
        # 後々単語の分散表現として使う
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f') # 今回は出力はone-hotで出てくるから入力と出力は同じ数！

        # レイヤー作成
        self.in_layer0 = MatMul(W_in) # contextsは2つしかないので，入力の重みは2つ準備
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # リストにまとめます
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers: # さっきの謎に[]でくくってるのはここでその意味を発揮　確か0から作るDL1だとここdictにしてた
            self.params += layer.params # 各レイヤーのものが[[w], [w], ]みたいに収納される！分かりやすい
            self.grads += layer.grads

        # 分散表現に使うもの
        self.words_vecs = W_in

    def forward(self, contexts, target):
        '''
        順伝播
        '''
        # 隠れ層の出力(ここで入力を2つに分けている！！)，バッチで入ってきても6 * 2 * 7　なので
        h0 = self.in_layer0.forward(contexts[:, 0]) # 1列目（前）
        h1 = self.in_layer1.forward(contexts[:, 1]) # 2列目（後ろ）
        # print(contexts)
        # print(contexts[:, 0])
        # sys.exit()
        # 2つの成分を足して2で割る，他にもやり方はある
        h = (h0 + h1) * 0.5
        # 出力のみ
        score = self.out_layer.forward(h)
        # lossも計算
        loss = self.loss_layer.forward(score, target)

        return loss

    def backward(self, dout=1): # 逆伝播は1からスタート　というかなんならいらない笑
        '''
        逆伝播
        '''
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)

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

