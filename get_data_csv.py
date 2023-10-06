import argparse
import csv
import logging
import os
import re
import opcodes_reentrancy_normal as orn
import get_opcode
import opcodes_reentrancy as or2


def get_bytecode(nowDir, writer, category):
    for home, dirs, files in os.walk(nowDir):
        for filename in files:
            row = []
            row.append(filename[:-4])
            nowfile = os.path.join(home, filename)
            if ".bin" in nowfile:
                f = open(nowfile, "r")
                bytecode = f.read()
                row.append(" ".join(re.findall(".{2}", bytecode)))
                row.append(category)
                writer.writerow(row)
                f.close()


def get_normal_dimension_opcode(nowDir, writer, category, feature2num):
    for home, dirs, files in os.walk(nowDir):
        for filename in files:
            row = []
            row.append(filename[:-4])
            nowfile = os.path.join(home, filename)
            if ".bin" in nowfile:
                f = open(nowfile, "r")
                bytecode = f.read()
                opcode = get_opcode.bytecode2opcode(bytecode)
                if opcode is None:
                    continue
                feature_code = ""
                for now_opcode in opcode:
                    feature_code += str(feature2num[orn.name2feature[now_opcode]]) + " "
                row.append(feature_code)
                row.append(category)
                writer.writerow(row)
                f.close()


def get_dimension_opcode(nowDir, writer, category, feature2num):
    for home, dirs, files in os.walk(nowDir):
        for filename in files:
            row = []
            row.append(filename[:-4])
            nowfile = os.path.join(home, filename)
            if ".bin" in nowfile:
                f = open(nowfile, "r")
                bytecode = f.read()
                opcode = get_opcode.bytecode2opcode(bytecode)
                if opcode is None:
                    continue
                feature_code = ""
                for now_opcode in opcode:
                    feature_code += str(feature2num[or2.name2feature[now_opcode]]) + " "
                row.append(feature_code)
                row.append(category)
                writer.writerow(row)
                f.close()


def main():
    f = open("normal_data.csv", "a", newline="")
    writer = csv.writer(f)
    feature = ['ADDRESS', 'OPCODE', 'CATEGORY']
    writer.writerow(feature)
    get_bytecode("./reentrancy_bin", writer, '1')
    get_bytecode("./novul_bin", writer, '0')
    f.close()


def main2():
    feature = set()
    feature2num = {}
    cnt = 0
    for i in orn.name2feature.values():
        feature.add(i)
    for i in feature:
        feature2num[i] = cnt
        cnt += 1
    f = open("normal_dimension_data.csv", "a", newline="")
    writer = csv.writer(f)
    feature = ['ADDRESS', 'OPCODE', 'CATEGORY']
    writer.writerow(feature)
    get_normal_dimension_opcode("./reentrancy_bin", writer, '1', feature2num)
    get_normal_dimension_opcode("./novul_bin", writer, '0', feature2num)
    f.close()


def main3():
    feature = set()
    feature2num = {}
    cnt = 0
    for i in or2.name2feature.values():
        feature.add(i)
    for i in feature:
        feature2num[i] = cnt
        cnt += 1
    f = open("dimension_data.csv", "a", newline="")
    writer = csv.writer(f)
    feature = ['ADDRESS', 'OPCODE', 'CATEGORY']
    writer.writerow(feature)
    get_dimension_opcode("./reentrancy_bin", writer, '1', feature2num)
    get_dimension_opcode("./novul_bin", writer, '0', feature2num)
    f.close()


if __name__ == '__main__':
    main()
    main2()
    main3()
