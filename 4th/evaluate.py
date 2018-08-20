# 評価用
import numpy as np
import pandas as pd

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

def load_file():
    # csvから読み込み
    word_vecs = pd.read_csv('word_vecs_data.csv', engine='python')
    word_vecs = word_vecs.values
    word_to_id = pd.read_csv('word_to_id_data.csv', header=None, engine='python')
    word_to_id = word_to_id.values
    id_to_word = pd.read_csv('id_to_word_data.csv', header=None, engine='python')
    id_to_word = id_to_word.values

    # 列削除(最初の行が読み込まれてしまうので)
    word_vecs = np.delete(word_vecs, 0, 1)
    word_to_id = np.delete(word_to_id, 0, 1)
    id_to_word = np.delete(id_to_word, 0, 1)

    # 辞書作成
    word_to_id = dict(zip(word_to_id[0, :], list(map(int, word_to_id[1, :]))))
    id_to_word = dict(zip(list(map(int, id_to_word[0, :])), id_to_word[1, :]))

    
    # print(word_to_id)
    # print(id_to_word)
    # print(word_vecs)

    return word_vecs, word_to_id, id_to_word

def main():
    word_vecs, word_to_id, id_to_word = load_file()

    querys = ['you', 'year', 'toyota']

    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=10)

if __name__ == '__main__':
    main()