# 標準ライブラリ系
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
import math

# NN関係
from figure import Formal_mul_ploter
from NN_trying import SimpleRnn # ネットワーク構成
from optimizer_trying import SGD # 最適化手法

# dataをreadするクラス 
class Data(): 
    def __init__(self, path):
        self.path = path

    def read(self):
        # dataの読み込み
        data = pd.read_csv(self.path, header=None, engine='python')
        data = data.values # numpyへ変換
        data_lines = data.shape[0] # データの総数（行数）

        # 各状態を格納
        self.data_dic = {}
        self.data_dic['date'] = data[:, 0]
        self.data_dic['ave_temp'] = data[:, 1]

        # 正規化
        self._min_max_normalization()

        # 文字列だけ加工します
        for i in range(len(self.data_dic['date'])):
            self.data_dic['date'][i] = datetime.datetime.strptime(self.data_dic['date'][i], '%Y/%m/%d')
        
        return self.data_dic

    def read_sample_data(self):
        total_size = 400
        # 各状態を格納
        self.data_dic = {}
        self.data_dic['date'] = [i for i in range(total_size)]
        self.data_dic['ave_temp'] = []

        # 何ステップか
        T = 15

        for i in range(total_size):
            self.data_dic['ave_temp'].append(math.sin((i/T) * 2 * math.pi))

        # 正規化
        # self._min_max_normalization()
        
        return self.data_dic

    
    def _min_max_normalization(self): # 最大最小正規化
        for data_name in ['ave_temp']:
            MAX = np.max(self.data_dic[data_name])
            MIN = np.min(self.data_dic[data_name])

            print('MAX = {0}'.format(MAX))
            print('MIN = {0}'.format(MIN))

            for k in range(len(self.data_dic['ave_temp'])):
                self.data_dic[data_name][k] = (self.data_dic[data_name][k] - MIN) / (MAX - MIN)


def main():
    # dataの読み込み
    path = 'data.csv'
    data_editer = Data(path)
    # sin波
    # data_dic = data_editer.read_sample_data()
    # 月別平均気温
    data_dic = data_editer.read()

    # data作成
    rate = 0.8
    data_size = len(data_dic['ave_temp'])
    train_size = int(data_size * rate)

    # Training_data
    x1 = np.array(data_dic['ave_temp'][:train_size-1], dtype='f')
    t1 = np.array(data_dic['ave_temp'][1:train_size], dtype='f')
    # x2 = data_dic['low_temp'][:train_size-1]
    # t2 = data_dic['low_temp'][1:train_size]
    print(x1)

    # test_data
    x_test = np.array(data_dic['ave_temp'][train_size:-1])
    t_test = np.array(data_dic['ave_temp'][train_size+1:])
    
    # plot
    x = data_dic['date']
    y = [data_dic['ave_temp']]
    x_label_name = 'date'
    y_label_name = 'temp'
    y_names = ['ave_temp']
    ploter = Formal_mul_ploter(x, y, x_label_name, y_label_name, y_names)
    ploter.mul_plot()

    # ハイパーパラメータの設定
    batch_size = 5 # バッチサイズ
    input_size = 1 # 入力の次元
    hidden_size = 20 # 隠れ層の大きさ
    output_size = 1 # 出力の次元
    time_size = 12  # Truncated BPTTの展開する時間サイズ，RNNのステップ数 # 20
    lr = 0.01 # 学習率 0.01
    max_epoch = 500 # 最大epoch
    data_size = len(x1)

    # 学習時に使用する変数
    max_iters = data_size // (batch_size * time_size) # 現実的に繰り返せる回数（データの数），ランダムに使うわけではない！！
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ave_loss_list = []

    # モデルの生成
    model = SimpleRnn(input_size, hidden_size, output_size)
    optimizer = SGD(lr)

    # ミニバッチの各サンプルの読み込み開始位置を計算
    jump = (data_size - 1) // batch_size
    offsets = [i * jump for i in range(batch_size)]

    print('max_iters = {0}'.format(max_iters))
    print('offsets = {0}'.format(offsets))
    
    for epoch in range(max_epoch):

        # offsets = np.random.randint(0, data_size-1, (batch_size))

        for iter in range(max_iters):
            # ミニバッチの取得
            batch_x = np.empty((batch_size, time_size, input_size), dtype='f')
            batch_t = np.empty((batch_size, time_size, input_size), dtype='f')

            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    batch_x[i, t, :] = x1[(offset + time_idx) % data_size] # 
                    batch_t[i, t, :] = t1[(offset + time_idx) % data_size] # 今回はbatchを作ってもバッチ×時間×1かも
                time_idx += 1

            # print('batch_t.shape =  {0}'.format(batch_t))
            # print('batch_x.shape =  {0}'.format(batch_x))
            # sys.exit()

            # 勾配を求め、パラメータを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # エポックごとにパープレキシティの評価
        ave_loss = total_loss / loss_count
        print('| epoch {0} | ave_loss {1}'.format(epoch, round(ave_loss, 5)))
        ave_loss_list.append(round(ave_loss, 5))
        total_loss, loss_count = 0, 0

    # グラフの描画
    x = np.arange(len(ave_loss_list))
    plt.plot(x, ave_loss_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('ave_loss')
    plt.show()

    # ネットワークを用いて次のデータを予測
    model.reset_state()
    input_x = np.array(x_test[:time_size].reshape(1, time_size, input_size), dtype='f') # 始めの入力を作成
    predict_y = []
    ans_t = []
    for i in range(len(t_test) - time_size):
        next_x = model.predict(input_x) # 次のものを予測
        # リスト化
        next_x = list(next_x.flatten())
        input_x = list(input_x.flatten())
        # 要素を削除して追加
        input_x.pop(0)
        input_x.append(next_x[-1])
        # print(t_test[time_size + i], next_x[-1])
        # a = input()
        predict_y.append(next_x[-1])
        ans_t.append(t_test[time_size-1 + i])
        
        input_x = np.array(input_x).reshape(1, time_size, input_size)

    plt.plot(range(len(t_test) - time_size), predict_y, label='pre')
    plt.plot(range(len(t_test) - time_size), ans_t, label='ans')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()