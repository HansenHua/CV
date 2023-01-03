  
# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import os
import csv
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img2vid_path = "../VehicleID/attribute/img2vid.txt"
model_attr_path = "../VehicleID/attribute/model_attr.txt"
pic_path = ""  # 填上图片根目录的路径
sample_path = "../VehicleID/train_test_split/3train16000.txt"
Vehicle_List = []
VehicleType_List = []
Train_List = []
Test_List = []


class Vehicle:
    pic_suffix = ".jpg"

    def __init__(self, _id, _type):
        self.vehicle_ID = _id
        self.type = _type
        self.picture = []  # 这个是图片名的列表。图片数量多，建议用一张读一张
        self.pic_num = 0


# 某一个类别的车的集合
class VehicleType:
    def __init__(self, _id):
        self.id = _id
        self.vehicle = []
        self.num = 0


def read_data(img2_path, model_path, v_list, vtype_list):
    f1 = open(img2_path, mode="r", encoding="utf-8")
    f2 = open(model_path, mode="r", encoding="utf-8")
    num = 0
    while True:
        s = f1.readline()
        if s == "":
            break
        tmp = s.split(" ")
        pno = tmp[0]
        vno = eval(tmp[1])
        if vno > num - 1:
            for i in range(vno - num + 1):
                v_list.append(Vehicle(vno + i, -1))
            num = vno + 1
        v_list[vno].picture.append(pno)
        v_list[vno].type = vno
        v_list[vno].pic_num += 1

    num = 0
    while True:
        s = f2.readline()
        if s == "":
            break
        tmp = s.split(" ")
        vno = eval(tmp[0])
        tno = eval(tmp[1])
        if tno > num - 1:
            for i in range(tno - num + 1):
                vtype_list.append(VehicleType(tno + i))
            num = tno + 1
        vtype_list[tno].vehicle.append(vno)
        v_list[vno].type = tno
        vtype_list[tno].num += 1


def make_sample(pos_num, neg_num, v_list, vtype_list, sample_path):  # 输入正负样本数量，生成样本
    neg_coefficient = 0.7  # 这是在负样本中同一类/不同车所占的比例
    result = []
    vt = len(vtype_list)
    for p in range(pos_num):
        r1 = random.randint(0, vt - 1)
        while vtype_list[r1].num == 0:
            r1 = random.randint(0, vt - 1)
        r2 = random.randint(0, vtype_list[r1].num - 1)
        vno = vtype_list[r1].vehicle[r2]
        r3 = random.randint(0, v_list[vno].pic_num - 1)
        r4 = random.randint(0, v_list[vno].pic_num - 1)
        result.append([v_list[vno].picture[r3], v_list[vno].picture[r4], 1])
    thr = neg_num * neg_coefficient
    for n in range(neg_num):
        r1 = random.randint(0, vt - 1)
        while vtype_list[r1].num == 0:
            r1 = random.randint(0, vt - 1)
        if n <= thr:
            r2 = random.randint(0, vtype_list[r1].num - 1)
            r3 = random.randint(0, vtype_list[r1].num - 1)
            while r3 == r2:
                r3 = random.randint(0, vtype_list[r1].num - 1)
            vno1 = vtype_list[r1].vehicle[r2]
            vno2 = vtype_list[r1].vehicle[r3]
            r4 = random.randint(0, v_list[vno1].pic_num - 1)
            r5 = random.randint(0, v_list[vno2].pic_num - 1)
            result.append([v_list[vno1].picture[r4], v_list[vno2].picture[r5], 0])
        else:
            r12 = random.randint(0, vt - 1)
            while r1 == r12 or vtype_list[r12].num == 0:
                r12 = random.randint(0, vt - 1)
            r2 = random.randint(0, vtype_list[r1].num - 1)
            r3 = random.randint(0, vtype_list[r12].num - 1)
            vno1 = vtype_list[r1].vehicle[r2]
            vno2 = vtype_list[r12].vehicle[r3]
            r4 = random.randint(0, v_list[vno1].pic_num - 1)
            r5 = random.randint(0, v_list[vno2].pic_num - 1)
            result.append([v_list[vno1].picture[r4], v_list[vno2].picture[r5], 0])
    if sample_path != "":
        f = open(sample_path, mode="w")
        for i in result:
            f.write("%s %s %d\n" % (i[0], i[1], i[2]))
    return result


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    read_data(img2vid_path, model_attr_path, Vehicle_List, VehicleType_List)
    print(len(Vehicle_List))
    print(len(VehicleType_List))
    res = make_sample(8000, 8000, Vehicle_List, VehicleType_List, sample_path)
    for r in res:
        print(r)
    print_hi('PyCharm')

