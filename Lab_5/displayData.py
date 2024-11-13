
import matplotlib.pyplot as plt
import numpy as np

def displayData(X, pred_str=None):
    # Determine the grid size for displaying data
    example_width = int(np.round(np.sqrt(X.shape[1])))
    example_height = int(X.shape[1] / example_width)

    display_rows = int(np.floor(np.sqrt(X.shape[0])))
    display_cols = int(np.ceil(X.shape[0] / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=(10, 10))

    if display_rows * display_cols == 1:
        ax_array = [ax_array]  # Convert to list for a single image

    # Iterate and display all images
    for i, ax in enumerate(np.ravel(ax_array)):
        if i < X.shape[0]:
            ax.imshow(X[i].reshape((example_width, example_height)).T, cmap='gray')
        ax.axis('off')

    if pred_str:
        plt.suptitle(pred_str)
    plt.show()