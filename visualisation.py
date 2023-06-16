"""Data visualization"""
from matplotlib import pyplot as plt


class Visualization:
    """Shows interactive plots with counter"""
    def __init__(self, img, num):
        plt.ion()
        self.img = img
        self.num = num

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.img)
        plt.title(label=f"Number of seeds: {self.num}")

        def onclick(event):
            if event.dblclick:
                if event.button == 1:
                    self.num += 1
                    ax.plot(event.xdata, event.ydata, 'co', markersize=15, mew=2, fillstyle="none")
                else:
                    self.num -= 1
                    ax.plot(event.xdata, event.ydata, 'rx', markersize=12, mew=2)
                plt.title(label=f"Number of seeds: {self.num}")

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)
        plt.close('all')
        print(f"[INFO] {self.num} seeds found by watershed algorythm with manual correction")
