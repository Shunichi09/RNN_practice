import sys
sys.path.append('..')
from NN_ch5 import SimpleRnnlm, SGD # , RnnlmTrainer
from common import ptb # dataset読み込む用
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ハイパーパラメータの設定
    batch_size = 10
    wordvec_size = 100 # Embeddingの大きさ
    hidden_size = 100 # 隠れ層の大きさ
    time_size = 5  # Truncated BPTTの展開する時間サイズ，RNNのステップ数
    lr = 0.1 # 学習率
    max_epoch = 100 # 最大epoch

    # 学習データの読み込み（データセットを小さくする）
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size] # コーパス小さくします
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]  # 入力，最後までは見ない
    ts = corpus[1:]  # 出力（教師ラベル），最初は飛ばす
    data_size = len(xs)
    print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

    # 学習時に使用する変数
    max_iters = data_size // (batch_size * time_size) # 現実的に繰り返せる回数（データの数），ランダムに使うわけではない！！，今回は99
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    # モデルの生成
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    # ミニバッチの各サンプルの読み込み開始位置を計算
    jump = (corpus_size - 1) // batch_size
    offsets = [i * jump for i in range(batch_size)]

    print('max_iters = {0}'.format(max_iters))
    print('offsets = {0}'.format(offsets))
    
    for epoch in range(max_epoch):
        for iter in range(max_iters):
            # ミニバッチの取得
            batch_x = np.empty((batch_size, time_size), dtype='i')
            batch_t = np.empty((batch_size, time_size), dtype='i')
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    batch_x[i, t] = xs[(offset + time_idx) % data_size] # 
                    batch_t[i, t] = ts[(offset + time_idx) % data_size] # 今回はbatchを作ってもバッチ×時間×1かも
                time_idx += 1
                
            print(time_idx)
            if time_idx > 200:
                sys.exit()

            # print('batch_t.shape =  {0}'.format(batch_t.shape))
            # print('batch_x.shape =  {0}'.format(batch_x.shape))

            # 勾配を求め、パラメータを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # エポックごとにパープレキシティの評価
        ppl = np.exp(total_loss / loss_count)
        print('| epoch %d | perplexity %.2f'
            % (epoch+1, ppl))
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

    # グラフの描画
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()

if __name__ == '__main__':
    main()
