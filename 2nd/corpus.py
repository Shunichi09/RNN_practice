# RNN基礎学習 
# 0から始めるdeeplearning 2

import numpy as np


def preprocess(text):
    text = text.lower() # すべて小文字に
    text = text.replace('.', ' .') # ピリオドがあった場合に最後の文字と分ける
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

def main():
    text = 'I say goodbye and you say hello.'

    print(preprocess(text))

if __name__ == '__main__':
    main()

