# 良く使うシリーズ
# chp3用の関数群
import numpy as np

def softmax(x):
    if x.ndim == 2: 
        x = x - x.max(axis=1, keepdims=True) # 形を守ったまま最大で構成して引き算する（計算結果がでないのを防ぐ）
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True) # これも形を守ったまま和を計算して各要素を割り算する
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y, t):
    if y.ndim == 1: # 1次元のとき（ばっちが1つのときってこと）行ベクトルに変形
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # エラーをとるのは，one-hot-vectorの正解部分だけです，マイナスとるのはもともとが小数点の話（softmaxの関係で）なので，マイナスになってしまうからです！
    return -1 * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # batchsizeで割り算して平均取ってます