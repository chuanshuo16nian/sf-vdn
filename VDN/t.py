from Fetch_3act import GameEnv
env = GameEnv()
import matplotlib.pyplot as plt
env.reset()
while True:
    act1 = input('act1:')
    act2 = input('act2:')
    r1, r2 = env.move(int(act1), int(act2))

    im = env.render_env()
    plt.imshow(im)
    plt.show(block=False)
    plt.pause(1)
    plt.clf()
    terminal = False
