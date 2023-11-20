import numpy as np
import pandas as pd
import kk.kk_datetime as kkd
import kk.kk_utils as kku


def main():
    path = "./rbBall.csv"
    fdata = pd.read_csv(path)
    odata = []

    for irow in fdata.values:
        tdata = []
        # 日期处理 (0-87位)
        dates = irow[1].split("/")
        iyear = int(dates[0])
        imonth = int(dates[1])
        iday = int(dates[2])
        zh_date = kkd.kk_date_ganzhi(iyear, imonth, iday, 19, 1)
        tdata = kku.kk_onehot(int(zh_date[0]), 10)
        tdata += kku.kk_onehot(int(zh_date[1]), 12)
        tdata += kku.kk_onehot(int(zh_date[2]), 10)
        tdata += kku.kk_onehot(int(zh_date[3]), 12)
        tdata += kku.kk_onehot(int(zh_date[4]), 10)
        tdata += kku.kk_onehot(int(zh_date[5]), 12)
        tdata += kku.kk_onehot(int(zh_date[6]), 10)
        tdata += kku.kk_onehot(int(zh_date[7]), 12)

        # 88-120 Red 处理
        t = kku.kk_onehot(irow[3] - 1, 33)
        for i in range(4, 9):
            t[irow[i] - 1] = 1
        tdata += t
        # 121-136 Blue 处理 (共137位)
        tdata += kku.kk_onehot(irow[9] - 1, 16)

        # 将 irow 转为 list 对象
        # tdata += irow[3:10].tolist()
        odata.append(tdata)
    odata = np.array(odata)
    np.save("./rbBall.npy", odata)


def split_data():
    fdata = np.load("./rbBall.npy")
    split_size = [1070, 100, 100000]
    split_size = [split_size[0], split_size[0]+split_size[1], split_size[0]+split_size[1]+split_size[2]]
    np.save("./rbBall_train.npy", fdata[:split_size[0]])
    np.save("./rbBall_val.npy", fdata[split_size[0]:split_size[1]])
    # np.save("./rbBall_test.npy", fdata[split_size[1]:split_size[2]])


if __name__ == "__main__":

    # main()

    split_data()

    fdata = np.load("./rbBall_train.npy")
    print(fdata.shape)
    # print(fdata[0:2, ...])
    # print(fdata[0:2, 88:])

    print(" - Over -")