import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def sreamplot_around_ep(dot_x, dot_y, title):
    plt.figure(figsize=(8, 8))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.streamplot(x, y, dot_x, dot_y, color='b', linewidth=1.5, density=1.5, maxlength=5)
    plt.plot(0, 0, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
    plt.title(title)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()

#sns.set_theme()

plt.rcParams.update({'font.size': 18, 'text.usetex': True})
x, y = np.meshgrid(np.linspace(-2, 2, 100),
                   np.linspace(-2, 2, 100))

dot_x = -4*x + 2*y
dot_y = x - 3*y
sreamplot_around_ep(dot_x, dot_y,'Stable node')


