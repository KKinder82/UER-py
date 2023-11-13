import pymysql
import os
import random
import pathlib as pl


def get_conn():
    db = pymysql.connect(host="127.0.0.1", user="root", password="root", database="bori_lcms")
    return db


def split(data_path):
    n_weights = [0.7,0.2,0.1]
    d_path = pl.Path(data_path)
    (d_path.parent / "split").mkdir(parents=True, exist_ok=True)
    d_path.parent.mkdir(parents=True, exist_ok=True)
    f_train = open(d_path.parent / "split" / "train.txt", "w", encoding="utf-8")
    f_test = open(d_path.parent / "split" / "test.txt", "w", encoding="utf-8")
    f_val = open(d_path.parent / "split" / "val.txt", "w", encoding="utf-8")
    _i = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            if line.startswith("label	texta"):
                f_train.write(line)
                f_test.write(line)
                f_val.write(line)
                continue
            r = random.random()
            if r < n_weights[0]:
                f_train.write(line)
            elif r < n_weights[0] + n_weights[1]:
                f_test.write(line)
            else:
                f_val.write(line)
            _i += 1
            print (f" >> {_i}")
    f_train.close()
    f_test.close()
    f_val.close()


def gen_corpus():
    db = get_conn()
    cursor = db.cursor()
    sql = "select topic from aidata_lltk order by rand()"
    cursor.execute(sql)
    data = cursor.fetchall()
    f = open("d:/aidata/dw_corpus.txt", "w", encoding="utf-8")
    i = 1
    for idata in data:
        text: str = idata[0]
        _len = len(text)
        t1 = text[:_len // 2]
        t2 = text[_len // 2:]
        f.writelines([t1, "\n", t2, "\n"])
        f.write("\n")
        print(f" >> {i}")
        i += 1
    f.close()
    cursor.close()
    db.close()
    print(" >> finished ")


def gen_data_train(count=100000):
    db = get_conn()
    cursor = db.cursor()
    sql = f"select knowledge_id, topic from aidata_lltk order by rand() LIMIT {count}"
    cursor.execute(sql)
    data = cursor.fetchall()
    f = open(f"d:/aidata/dw_train_cls.txt", "w", encoding="utf-8")
    i = 1
    for idata in data:
        cls = str(idata[0])
        text: str = idata[1]
        text = text.replace("\n", "")
        f.writelines([cls, "\t", text, "\n"])
        print( f" >> {i}")
        i += 1
    f.close()
    cursor.close()
    db.close()
    print(" >> finished ")


def main():
    # gen_corpus()
    # gen_data_train()
    split("d:/aidata/dw_train_cls.txt")


if __name__ == "__main__":
    main()