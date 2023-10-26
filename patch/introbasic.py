def getbasic(call, patch, contr, mode):
    """
        patch: patch代码
        contr: 合约相关信息
    """
    ret_basic = {}
    introductions = set()
    if mode == 0:
        for pos in patch[call]:
            for i in patch[call][pos]:
                introductions.add(i)
    introductions.add(call)
    introductions = sorted(introductions)
    j = 0
    for x, y in enumerate(contr.blocks):
        if x == len(contr.blocks):
            while j < len(introductions):
                ret_basic[introductions[j]] = y
                if introductions[j] == call:
                    callbasic = y
            break
        while j < len(introductions) and introductions[j] < contr.blocks[x + 1].offset:
            ret_basic[introductions[j]] = y
            if introductions[j] == call:
                callbasic = y
            j += 1
    basics = ret_basic.values()


    return callbasic, ret_basic
