import re
import os
import opcodes
def extract_metadata(bytecode):
    try:
        return re.search(r"a165627a7a72305820\S{64}0029$", bytecode).group()
    except:
        try:
            return re.search(r"a26[4-5].*003[2-3]", bytecode).group()
        except:
            return ""


def get_runtime_code(bytecode):
    metadata = extract_metadata(bytecode)
    ret = bytecode
    if metadata:
        try:
            ret = re.search(r"396000f300.*a165627a7a72305820\S{64}0029$", bytecode).group().replace("396000f300", "")
        except:
            try:
                ret =  re.search(r"396000f3fe.*a26[4-5].*003[2-3]$", bytecode).group().replace("396000f3fe", "")
            except:
                pass
        ret = ret.replace(metadata, "")
    else:
        try:
            deployment_bytecode = re.search(r".*396000f300", bytecode).group()
            if len(re.compile("396000f300").findall(deployment_bytecode)) == 1:
                return deployment_bytecode
            else:
                return deployment_bytecode.split("396000f300")[0] + "396000f300"
        except:
            try:
                deployment_bytecode = re.search(r".*396000f3fe", bytecode).group()
                if len(re.compile("396000f3fe").findall(deployment_bytecode)) == 1:
                    return deployment_bytecode
                else:
                    return deployment_bytecode.split("396000f3fe")[0] + "396000f3fe"
            except:
                deployment_bytecode = ""
        ret = bytecode.replace(deployment_bytecode, "")
    return ret


def bytecode2opcode(bytecode):
    pc = 0
    raw_bytecode = bytecode.strip()
    opcode = []
    try:
        runtime_bytecode = bytes.fromhex(get_runtime_code(raw_bytecode))
    except:
        return None
    while pc < len(runtime_bytecode):
        op = runtime_bytecode[pc]
        if op not in opcodes.listing:
            raise KeyError('Invalid op. op: {:#x}, offset: {:#x}'.format(op, pc))
        name = opcodes.listing[op][0]
        opcode.append(name)
        size = opcodes.operand_size(op)
        pc += size + 1
    return opcode


def get_bytecode(nowDir):
    bytecodes = []
    for home, dirs, files in os.walk(nowDir):
        for filename in files:
            nowfile = os.path.join(home, filename)
            if ".bin" in nowfile:
                f = open(nowfile, "r")
                bytecode = f.read()
                bytecodes.append(bytecode)
                f.close()
    return bytecodes