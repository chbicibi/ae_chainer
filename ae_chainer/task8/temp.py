
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc


def __test__():
    from scipy import stats
    fig, ax = plt.subplots()
    x = np.arange(0, 1, 0.01).reshape((1, -1))

    colors = [(0, '#ff0000'), (0.5, '#ffffff'), (1, '#00ff00')]
    # colors = [(0, '#0000ff'), (0.5, '#ffffff'), (1, '#ff0000')]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', colors)

    y = stats.norm.pdf(x)
    p = ax.imshow(x, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(p, orientation='horizontal')
    # cbar.set_label('', size=20)
    plt.show()


__test__()
