import tensorflow as tf
import numpy as np
a = []
a.append([1,1,1])
a.append([2,2,2])
a.append([3,3,3])
c = []
for i in range(3):
    c.append(a)
d = [[1],[0],[0]]
e = [[1],[0],[0]]
f = [[0],[0],[1]]
g=[]
g.append(d)
g.append(e)
g.append(f)
s = tf.Session()
print(s.run(tf.reduce_sum(tf.multiply(c,g),reduction_indices=1)))

# g = np.zeros([3,1])
# print(g)
# a = tf.placeholder(tf.int8,shape=[None, 3])
# b = tf.reshape(a, [-1, 3, 1])
# d = [[1,1,1],
#      [2,2,2]]
# print(s.run(b,feed_dict={a:d}))