# RNN基礎学習 
# 0から始めるdeeplearning 2
# chap3

import sys
sys.path.append('..') # これで上のディレクトリをpathに追加
import numpy as np
import NN_ch3

# 下準備に使うものたち

def preprocess(text):
    '''
    # 文字を分割して，単語ごとにid化，さらに単語のidののリストを返す
    # コーパス作成（要は辞書）
    # corpus = ['id_1', 'id_2', 'id_1'] のように，テキストのデータ順にidが割り振られらもの！
    '''
    text = text.lower() # すべて小文字に
    text = text.replace('.', ' .') # ピリオドがあった場合に最後の文字と分ける
    text = text.replace(',', ' ,') # カンマがあった場合に最後の文字と分ける
    words = text.split(' ')

    word_to_id = {} # wordからidを呼び出すやつ
    id_to_word = {} # idでwordを呼び出すやつ

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id) # どこに作るかだけね　-1されるから
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words]) # 単語のidのリスト化

    return corpus, word_to_id, id_to_word

def create_contexts_target(corpus, window_size=1):
    '''
    # CROWモデルに使用するためにコーパスから入力とターゲットを作る
    '''
    target = corpus[window_size:-window_size] # widnowsize分だけ両側は排除しておく（2つを入力にいれることができなくなるため）
    contexts = [] # 作成するcontextの組み合わせ

    for idx in range(window_size, len(corpus)-window_size): # index
        cs = [] # これがcontextsを構成する1要素になる
        for t in range(-window_size, window_size+1): # 取り出したcorpusからどこをcontextsとして取り出すのかを見ている/-1からはじまります
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs) # 作成したものを本体に追加
    
    return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size): # 個人的にはcorpusという表現はいまいちかと，単純にtargetとcontexts
    '''one-hot表現への変換
    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1: # 次元が1の場合，たぶんこれはtarget用
        one_hot = np.zeros((N, vocab_size), dtype=np.int32) # zerosで初期化
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1 # 必要なところだけ1にする

    elif corpus.ndim == 2:# 次元が2の場合，たぶんこれはcontexts用
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):# zerosで初期化
            for idx_1, word_id in enumerate(word_ids):# 必要なところだけ1にする
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

## ここからNNのクラス（同フォルダの別pyに記載）

from NN_ch3 import SimpleCBOW, Adam, Trainer

def main():
    # 準備
    # これは全部そのままの意味
    window_size = 1 # どれくらいの長さ見るか
    hidden_size = 5 # 隠れ層の数
    batch_size = 3 # バッチサイズ
    max_epoch = 1000
    
    # テキストデータ
    text = 'You say goodbye and I say hello'
    # コーパスと辞書作成
    corpus, word_to_id, id_to_word = preprocess(text)
    # GROWに使うデータセット作成
    contexts, target = create_contexts_target(corpus, window_size=1)
    # 語彙数
    vocab_size = len(word_to_id)

    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    
    # 使うモデル
    model = SimpleCBOW(vocab_size, hidden_size) # 入力と出力が同じなので（大きさは語彙数になる）
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # trainer
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

if __name__ == '__main__':
    main()


