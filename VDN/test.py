import tensorflow as tf

a = tf.constant([[1,2,3,4,5],
                 [6,7,8,9,0]])
# a = tf.reshape(a,[-1])
# l = []
# for i in range(5):
#     l.append(a)
# l = tf.transpose(l, [1, 0])
# l = tf.reshape(l ,[-1, 25])
#tmp1 = tf.transpose(l, [0,2, 1])
#tmp2 = tf.reshape(tmp1,[-1, 25])
a = tf.concat([a,a], 1)

#c = tf.concat([a,b],0)
s = tf.Session()
print(s.run(a))
#print(s.run(tmp1))
#print(s.run(tmp2))