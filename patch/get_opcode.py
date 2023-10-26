import argparse
import csv
import logging
import os
import re
import json
import time
import signal
import opcodes


def remove_swarm_hash(evm):
    pattern = re.compile(r'a165627a7a72305820\w{64}0029$', re.A)
    if pattern.search(evm):
        evm = evm[:-86]
    return evm

def bytecode2opcode(bytecode):
    pc = 0
    raw_evm = bytecode.strip()
    opcode = []
    try:
        evm = bytes.fromhex(remove_swarm_hash(raw_evm))
    except:
        return None
    while pc < len(evm):
        op = evm[pc]
        if op not in opcodes.listing:
            raise KeyError('Invalid op. op: {:#x}, offset: {:#x}'.format(op, pc))
        name = opcodes.listing[op][0]
        opcode.append('{0} {1}:{2}'.format(hex(pc), pc, name))
        size = opcodes.operand_size(op)
        for i in range(0, size):
            opcode[-1] += " {0}".format(hex(evm[pc + i + 1]))
        pc += size + 1
    return opcode

def bytecodes2opcodes(bytecodes):
    ret_opcodes = []
    mark_opcode = ""
    llen = 30000
    for bytecode in bytecodes:
        pc = 0
        now_len = 0
        raw_evm = bytecode.strip()
        opcode = ""
        try:
            evm = bytes.fromhex(remove_swarm_hash(raw_evm))
        except:
            continue
        while pc < len(evm):
            op = evm[pc]
            if op not in opcodes.listing:
                raise KeyError('Invalid op. op: {:#x}, offset: {:#x}'.format(op, pc))
            name = opcodes.listing[op][0]
            opcode += name + "\n"
            size = opcodes.operand_size(op)
            pc += size + 1
            now_len += 1
        if "SSTORE" in opcode:
            if now_len < llen:
                llen = now_len
                mark_opcode = opcode

    return ret_opcodes


def get_bytecode(nowDir):
    bytecodes = []
    for home, dirs, files in os.walk(nowDir):
        for filename in files:
            if filename != 'patchedReentrancy2.bin':
                continue
            nowfile = os.path.join(home, filename)
            if ".bin" in nowfile:
                f = open(nowfile, "r")
                bytecode = f.read()
                bytecodes.append(bytecode)
                f.close()
    return bytecodes


def main():
    bytecodes = get_bytecode(".")

    ret = bytecode2opcode(bytecodes[0])
    for i in ret:
        print(i)

    # f = open("reentrancy_data.csv", "a", newline="")
    # writer = csv.writer(f)
    # writer.writerow(opcodes.feature)
    # for opcode in ret_opcodes:
    #     record = {'OP_LENGTH': len(opcode), 'OPER': 0, 'COMP': 0, "ADDRESS": 0, "VALUE": 0,
    #               "STACK&MEMORY": 0, "STORAGE": 0, "JUMP": 0, "LOG": 0, "CALL": 0, "RETURN": 0, "LABEL": 1}
    #     for op in opcode:
    #         record[op] += 1
    #     write_row = []
    #     for i in record:
    #         write_row.append(record[i])
    #     writer.writerow(write_row)


    # bytecodes = get_bytecode("../dataset/novul_bin")
    # ret_opcodes = bytecodes2opcodes(bytecodes)
    # for opcode in ret_opcodes:
    #     record = {'OP_LENGTH': len(opcode), 'OPER': 0, 'COMP': 0, "ADDRESS": 0, "VALUE": 0,
    #               "STACK&MEMORY": 0, "STORAGE": 0, "JUMP": 0, "LOG": 0, "CALL": 0, "RETURN": 0, "LABEL": 0}
    #     for op in opcode:
    #         record[op] += 1
    #     write_row = []
    #     for i in record:
    #         write_row.append(record[i])
    #     writer.writerow(write_row)
    #
    # f.close()



if __name__ == '__main__':
    main()


