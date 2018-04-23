# import tensorflow as tf
# import numpy as np
# a = []
# a.append([1,1,1])
# a.append([2,2,2])
# a.append([3,3,3])
# c = []
# for i in range(3):
#     c.append(a)
# d = [[1],[0],[0]]
# e = [[1],[0],[0]]
# f = [[0],[0],[1]]
# g=[]
# g.append(d)
# g.append(e)
# g.append(f)
# s = tf.Session()
# print(s.run(tf.reduce_sum(tf.multiply(c,g),reduction_indices=1)))
#
# # g = np.zeros([3,1])
# # print(g)
# # a = tf.placeholder(tf.int8,shape=[None, 3])
# # b = tf.reshape(a, [-1, 3, 1])
# # d = [[1,1,1],
# #      [2,2,2]]
# # print(s.run(b,feed_dict={a:d}))
import pandas as pd
WIN = False
def save_reward(l):
    path = '/home/admin1/zp/test.csv'
    name = ['s', 'a', 'r', 'ns', 'terminal']
    file = pd.DataFrame(columns=name, data=l)
    file.to_csv(path)


def read_reward():
    path = '/home/admin1/zp/test.csv'
    try:
        file = open(path, 'r', encoding="utf-8")
        context = file.read()
        list_result = context.split("\n")
        length = len(list_result)
        for i in range(length):
            list_result[i] = list_result[i].split(",")
        return list_result
    except Exception:
        print('Failed to read rd!')
    finally:
        file.close()
l = [[[1,2],[22,23],3,[5,5],False],[[1,2],[22,23],3,[5,5],False]]
save_reward(l)
