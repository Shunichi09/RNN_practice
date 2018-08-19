# RNN基礎学習 
# 0から始めるdeeplearning 2
# chap3

import sys
sys.path.append('..') # これで上のディレクトリをpathに追加
import numpy as np
from NN_ch4 import CBOW, Adam, Trainer
from common import ptb # dataset読み込む用
import pandas as pd

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
    target = corpus[window_size:-window_size] # widnowsize分だけ両側は排除しておく（前後2つを入力にいれることができなくなるため）
    contexts = [] # 作成するcontextの組み合わせ

    for idx in range(window_size, len(corpus)-window_size): # index
        cs = [] # これがcontextsを構成する1要素になる
        for t in range(-window_size, window_size+1): # 取り出したcorpusからどこをcontextsとして取り出すのかを見ている/-1からはじまります
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs) # 作成したものを本体に追加
    
    return np.array(contexts), np.array(target)


def main():
    # 準備
    # これは全部そのままの意味
    window_size = 5 # どれくらいの長さ見るか
    hidden_size = 10 # 隠れ層の数(hの列数)
    batch_size = 100 # バッチサイズ
    max_epoch = 10
    
    # コーパスと辞書作成
    corpus, word_to_id, id_to_word = ptb.load_data('train')

    # GROWに使うデータセット作成
    contexts, target = create_contexts_target(corpus, window_size=1) # 5
    # 語彙数
    vocab_size = len(word_to_id)

    print('vocab_size is {0}'.format(vocab_size))
    
    # 使うモデル
    model = CBOW(vocab_size, hidden_size, window_size, corpus) # 入力と出力が同じなので（大きさは語彙数になる）
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # trainer
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    # 単語の分散を見てみる
    word_vecs = model.word_vecs

    # wordvec用
    word_vecs_pandas = pd.DataFrame(word_vecs)

    # word_to_id 用
    word_to_id_pandas = pd.io.json.json_normalize(word_to_id)

    # id_to_word　用
    id_to_word_pandas = pd.io.json.json_normalize(id_to_word)

    name_list = ('word_vecs', 'word_to_id', 'id_to_word')

    for i, item in enumerate(word_vecs_pandas, word_to_id_pandas, id_to_word_pandas):
        item.to_csv(name_list[i] +'_data.csv')

if __name__ == '__main__':
    main()


