import argparse
import datetime
import json
import os
import re
import access
import patch
import introbasic
import generate


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
                ret = re.search(r"396000f3fe.*a26[4-5].*003[2-3]$", bytecode).group().replace("396000f3fe", "")
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
                #print("Error: Unknown bytecode format:", bytecode)
        ret = bytecode.replace(deployment_bytecode, "")
    return ret


def resolve_metadata(data):
    reentrancy = set()
    if 'Reentrancy' in data:
        for vul in data['Reentrancy']:
            call = vul['call']
            sstore = vul['sstore']
            mode = vul['mode']
            reentrancy.add((call, sstore, mode))
    return reentrancy, mode, call


def main():
    # parser = argparse.ArgumentParser(description='Patcher')
    # parser.add_argument('-b', '--bytecode',
    #                     type=str, required=True, help='bytecode file')
    # parser.add_argument('-m', '--metadata',
    #                     type=str, required=True, help='Vulnerability metadata file (JSON)')
    # parser.add_argument('-o', '--output',
    #                     type=str, required=True, help='Patched EVM bytecode file (HEX)')
    #
    # args = parser.parse_args()

    # 读取字节码并去除部署代码和auxdata

    #0xf015c35649c82f5467c9c74b7f28ee67665aad68
    nowDir = "./dataset/horus/metadata_withmode"
    for home, dirs, files in os.walk(nowDir):
        for filename in files:
            if ".json" in filename:
                open_path = nowDir + "/" + filename

                print(filename)
                with open(open_path) as f:
                    metadata = json.load(f)
                    reentrancy, mode, call = resolve_metadata(metadata)
                    with open("./dataset/horus/reentrancy/" + filename[:-4] + "bin") as f:
                        bytecode = f.read().strip()
                        bytecode = get_runtime_code(bytecode)
                        try:
                            bytecode = bytes.fromhex(get_runtime_code(bytecode))
                        except:
                            print("false")
                print(reentrancy)

                report = {
                    'Error': None,
                    'Timeout': False,
                    'Finished': False,
                    'Time': None,
                    'Reentrancy': [],
                    'IntegerBugs': [],
                }
                contr, dfg, cfg, endpc = access.initialize(bytecode)
                trace, free_storage = access.analyze(contr, dfg, cfg)
                try:
                    call, patches = patch.execute(dfg, trace, reentrancy, report)
                except:
                    patches = {}

                if len(patches) == 0:
                    mode = 1
                #0x01f8c4e3fa3edeb29e514cba738d87ce8c091d3f
                open_path = "./dataset/horus/patch/" + filename[:-4] + "bin"
                callbasic, retbasic = introbasic.getbasic(call, patches, contr, mode)
                bytecode = generate.getbytecode(contr, call, callbasic, dfg, patches, retbasic, endpc, mode, free_storage[0])
                with open(open_path, 'w') as f:
                    f.write(bytecode.hex())
    # except Exception as ex:
    #     # open_path = "simple_dao_patched.err"
    #     # with open(open_path, 'w') as f:
    #     #     f.write("ex")
    #     print(ex)


    # nowDir = "./dataset/smartbugs/metadata/reentrancy/"
    #
    # time1 = datetime.datetime.now()
    # for home, dirs, files in os.walk(nowDir):
    #     for filename in files:
    #         if ".json" in filename:
    #             open_path = nowDir + filename
    #             with open(open_path) as f:
    #                 metadata = json.load(f)
    #                 reentrancy = resolve_metadata(metadata)
    #             open_path = "./dataset/smartbugs/reentrancy/" + filename[:-5] + ".bin"
    #             with open(open_path) as f:
    #                 bytecode = f.read().strip()
    #                 bytecode = get_runtime_code(bytecode)
    #                 try:
    #                     bytecode = bytes.fromhex(get_runtime_code(bytecode))
    #                 except:
    #                     print(filename)
    #
    #
    #                 # 读取重入的漏洞相关信息
    #                 # with open(args.metadata) as f:
    #                 #     metadata = json.load(f)
    #                 #     reentrancy = resolve_metadata(metadata)
    #
    #
    #                 report = {
    #                     'Error': None,
    #                     'Timeout': False,
    #                     'Finished': False,
    #                     'Time': None,
    #                     'Reentrancy': [],
    #                     'IntegerBugs': [],
    #                 }
    #                 try:
    #                     contr, dfg, cfg, endpc = access.initialize(bytecode)
    #                     trace = access.analyze(contr, dfg, cfg)
    #                     call, patches = patch.execute(dfg, trace, reentrancy, report)
    #                     # miscellany存储的是整数溢出漏洞，先不用管
    #
    #                     open_path = "./dataset/smartbugs/contrast_patched_reentrancy/" + filename[:-5] + "patched.bin"
    #                     callbasic, retbasic = introbasic.getbasic(call, patches, contr, filename[:-5])
    #                     mode = 2
    #                     bytecode = generate.getbytecode(contr, call, callbasic, dfg, patches, retbasic, endpc, mode)
    #                     # with open(open_path, 'w') as f:
    #                     #     f.write(bytecode.hex())
    #                 except Exception as ex:
    #                     open_path = "./dataset/smartbugs/contrast_patched_reentrancy/" + filename[:-5] + "patched.err"
    #                     with open(open_path, 'w') as f:
    #                         f.write("ex")
    # time2 = datetime.datetime.now()
    # print(time2 - time1)




if __name__ == '__main__':
    main()
