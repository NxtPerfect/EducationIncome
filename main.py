import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Process
from time import perf_counter

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
    plt.tight_layout()
    plt.show()

def showPlotDebug(data):
    # Sort results by biggest
    data = data[data["Education level"] != "Total, all education levels"]
    plt.title("Testing")
    print("Loading data...")
    t1_start = perf_counter()
    plt.bar(data["Education level"][:100_000], data["Both Sexes"][:100_000])
    t1_stop = perf_counter()
    print(f'Finished loading in{(t1_stop-t1_start): .2f} seconds.')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # process = Process(target=loadCSV, args=("./data/incom vs education.csv",))
    # process.start()
    # process.join()
    loadCSVDebug(DATA)
