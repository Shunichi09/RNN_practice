# RNN基礎学習 
# 0から始めるdeeplearning 2

import numpy as np

# コーパス作成（要は辞書）
# corpus = ['id_1', 'id_2', 'id_1'] のように，テキストのデータ順にidが割り振られらもの！

def preprocess(text): # 文字を分割して，単語ごとにid化，さらに単語のidののリストを返す
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

# 共起行列(id（基準の行列） ×　id) vocab_size は単純にidの数 id順に作成
def creat_co_matrix(corpus, vocab_size, window_size=1): # 共起行列の作成 # corpusの長さじゃだめ！，必要なのは辞書の長さ
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32) # これでテーブルを作っている

    for idx, word_id in enumerate(corpus): # インデックスと中身
        for i in range(1, window_size + 1): # windowsizeでどこを見てるか見てる
            left_idx = idx - 1 # これが左見てる
            right_idx = idx + 1 # これで右見てる（インデックス）

            if left_idx >= 0: # 左側が0より小さいとまずいので
                left_word_id = corpus[left_idx] # これで，コーパスで見たときの文字の左のidがわかるので
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size: # corpus(文字列のサイズ)よりは小さく！ 
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

# 相関係数
def cos_similarity(x, y, eps=1e-8): # ベクトル入れて帰ってくる，相互相関と同じ
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)

    return np.dot(nx, ny) # ベクトル同士なので内積になります

# 並び替えする(ここでいうとqueryに近い順)
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # バグ処理
    if query not in word_to_id:
        print('{0} is not found'.format(query))
        return # バグなので早ぬけ

    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id] # コーパスをもとに作られているのでidで（行）指定

    vocab_size = len(id_to_word) # vocabの数，何個辞書が単語持っているか
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size): # 各idのものと比較するだけです
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort(): # 並び替えてくれる（しかも返り値は，indexという最強関数）
        if id_to_word[i] == query:
            # 自分ならpass
            continue
        
        print('{0}, {1}' .format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top: # 5番目まで出力！！
            return

def main():
    text = 'You say goodbye and I say hello.' #  I want to say hello but you say goodbye.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)

    vocab_size = len(word_to_id) # 辞書の長さ
    C = creat_co_matrix(corpus, vocab_size) # 共起行列作成

    c0 = C[word_to_id['you']]
    c1 = C[word_to_id['i']]

    print(cos_similarity(c0, c1))

    most_similar('you', word_to_id, id_to_word, C)

if __name__ == '__main__':
    main()