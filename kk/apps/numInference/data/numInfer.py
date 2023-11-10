import numpy as np
import pandas as pd
import kk.kk_datetime as kkd
import kk.kk_utils as kku


def main():
    path = "./numInfer.csv"
    fdata = pd.read_csv(path)
    odata = []

    for irow in fdata.values:
        tdata = irow[1:9].tolist()
        odata.append(tdata)
    odata = np.array(odata)
    np.save("./numInfer.npy", odata)


def split_data():
    fdata = np.load("./numInfer.npy")
    np.save("./numInfer_train.npy", fdata[:25])
    np.save("./numInfer_val.npy", fdata[25:25+10])
    np.save("./numInfer_test.npy", fdata[35:])


if __name__ == "__main__":

    main()

    fdata = np.load("./numInfer.npy")
    print(fdata.shape)

    split_data()

    print(" - Over -")