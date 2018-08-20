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

    print(word_to_id)


    return word_vecs, word_to_id, id_to_word

def main():
    word_vecs, word_to_id, id_to_word = load_file()

    querys = ['you', 'year']

    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

if __name__ == '__main__':
    main()


'''
import numpy as np
a,b,c = np.arange(10), np.arange(10)*2, np.arange(10)*3

#-> a: [0 1 2 3 4 5 6 7 8 9]
#-> b: [ 0  2  4  6  8 10 12 14 16 18]
#-> c: [ 0  3  6  9 12 15 18 21 24 27]

_bc = zip(b,c)

result = dict(zip(a,_bc))
#-> {0: (0, 0), 1: (2, 3), 2: (4, 6), 3: (6, 9), 4: (8, 12), 5: (10, 15), 6: (12, 18), 7: (14, 21), 8: (16, 24), 9: (18, 27)}

# result = dict(zip(a,zip(b,c)))でもイイ。
'''