import tensorflow as tf

a = tf.constant([1,2,3,4,5])
b = tf.constant([4, 5, 6])
l = []
for i in range(5):
    l.append(a)
tmp1 = tf.transpose(l, [1, 0])
tmp1 = tf.reshape(tmp1,[-1])

c = tf.concat([a,b],0)
s = tf.Session()
print(s.run(tmp1))