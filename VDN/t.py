import matplotlib.pyplot as plt
import numpy as np

from Fetch import GameEnv

env = GameEnv()
env.reset()

temp = env.render_env()

i = 0
show = False
t = 0
# fig = plt.figure()
# f1 = fig.add_subplot(121)
# f2 = fig.add_subplot(122)
while True:
    if show:
        # print(env.agent1.y, env.agent1.x, env.agent2.y, env.agent2.x)
        # print(env.get_index())
        t1,t2 = env.get_states()
        print(t1.shape)
        plt.imshow(t1)
        # f2.imshow(t2)

        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
        t += 1
        if not t % 100:
            show = False
    action1 = np.random.randint(8)
    action2 = np.random.randint(8)
    i += 1
    r1, r2 = env.move(action1, action2)
    if r1 or r2:
        show = True
        print(i, 'r1:', r1, 'r2:', r2)
