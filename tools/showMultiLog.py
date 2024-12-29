import os

from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

def main():
    # load log data
    exp = "E025_RGB_ManifoldADD"
    in_path = os.path.join("/home/d409/SatellitePoseEstimation/result/E0", exp, "log")

    event_data = event_accumulator.EventAccumulator(in_path)  # a python interfacefor loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys[:])  # my first column is training loss periteration, so I abandon it
    for key in keys:
        # print(key)
        df[key] = pd.DataFrame(event_data.Scalars(key)).value
    lossTrain = df['loss'].to_numpy()
    lossVal = df['val_loss'].to_numpy()
    lossVal = lossVal[~np.isnan(lossVal)]

    fig, ax = plt.subplots(figsize=(5, 5))
    xVal = np.arange(5, len(lossVal)*5+5, 5)
    plt.plot(xVal, lossVal, color='red', label='Val')

    xTrain = np.arange(0, len(lossVal)*5, len(lossVal)*5./len(lossTrain))
    plt.plot(xTrain, lossTrain, color='blue', label='Train')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 0.6)
    plt.xlim(0, len(lossVal)*5+6)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Average Distance Error', fontsize=14)
    plt.legend(fontsize=18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    plt.show()

if __name__ == '__main__':
    main()
