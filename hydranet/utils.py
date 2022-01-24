from typing import List, Tuple
import matplotlib.pyplot as plt


def plot_images(
    images: List = [], titles: List = [], figure_size: Tuple[int, int] = (12, 12)
):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        if len(titles) != 0:
            fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        else:
            fig.add_subplot(1, len(images), i + 1)
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()
