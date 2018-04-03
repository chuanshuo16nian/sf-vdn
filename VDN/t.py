import matplotlib.pyplot as plt
import numpy as np
from PIL import  Image
from Fetch import GameEnv
from pylab import *
# env = GameEnv()
# env.reset()
# temp = env.render_env()
# t1, t2 = env.get_states()
# im = Image.fromarray(np.uint8(t1 * 255))
# im = im.convert('L')
# arr = array(im)
# print(arr)
# arr = np.reshape(arr,[-1])
# print(arr)
# im.show()
# def to_gray(state):
#     im = Image.fromarray(np.uint8(state * 255))
#     im = im.convert('L')
#     return array(im)
# i = 0
# show = False
# t = 0
# # fig = plt.figure()
# # f1 = fig.add_subplot(121)
# # f2 = fig.add_subplot(122)
# while True:
#     if show:
#         # print(env.agent1.y, env.agent1.x, env.agent2.y, env.agent2.x)
#         # print(env.get_index())
#         t1,t2 = env.get_states()
#         im = Image.fromarray(np.uint8(t1 * 255))
#         im = im.convert('L')
#         print(t1.shape)
#         # plt.imshow(im)
#         # # f2.imshow(t2)
#         #
#         # plt.show(block=False)
#         # plt.pause(0.01)
#         # plt.clf()
#         t += 1
#         if not t % 100:
#             show = False
#     action1 = np.random.randint(8)
#     action2 = np.random.randint(8)
#     i += 1
#     r1, r2 = env.move(action1, action2)
#     if r1 or r2:
#         show = True
#         print(i, 'r1:', r1, 'r2:', r2)
print(63%8)