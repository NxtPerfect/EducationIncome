import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Process
from time import perf_counter
from collections import defaultdict

DATA = "./data/incom vs education.csv"

def loadCSV(path: str):
    data = pd.read_csv(path)
    showPlot(data)

def loadCSVDebug(path: str):
    dtypes = {
        "Education level": str,  # Assuming education level is categorical
        "Both Sexes": float  # Assuming average wages are floats
    }
    data = pd.read_csv(path, dtype=dtypes)
    showPlotDebug(data)

def showPlot(data):
    plt.title("Testing")
    print("Loading data...")
    plt.bar(data["Education level"], data["Both Sexes"])
    print("Finished loading.")
    plt.xticks(rotation=45)
    plt.show()

def showPlotDebug(data):
    plt.figure(figsize=(10,6))
    # Sort results by biggest
    data = data[data["Education level"] != "Total, all education levels"]
    plt.title("Education level vs Income")
    data = data.groupby("Education level", as_index=False)['Both Sexes'].mean().reset_index()
    data = data.sort_values(by=["Both Sexes"], ascending=False)
    # If i sum up the elements before the plot, it should be faster

    print("Loading data...")
    t1_start = perf_counter()
    barlist = plt.bar(data["Education level"], data["Both Sexes"])

    for i, item in enumerate(barlist):
        if not i % 3:
            item.set_color('b')
            continue
        if not i % 4:
            item.set_color('g')
            continue
        if not i % 2:
            item.set_color('r')
            continue
    t1_stop = perf_counter()
    print(f'Finished loading in{(t1_stop-t1_start): .2f} seconds.')
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # process = Process(target=loadCSV, args=("./data/incom vs education.csv",))
    # process.start()
    # process.join()
    loadCSVDebug(DATA)
