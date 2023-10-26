import opcodes
def instruction_to_bytecode(instr):
    """
        Convert instruction to bytecode
    """
    op = instr.op
    evm = bytearray(op.to_bytes(1, byteorder='big'))
    if opcodes.is_push(op):
        arg = instr.arg
        size = opcodes.operand_size(op)
        evm += bytearray(arg.to_bytes(size, byteorder='big'))
    return evm


def instruction_to_bytecodeINVALID(instr):
    """
        将该指令用invalid替代
    """
    op = instr.op
    # 取0x0C做invalid值
    retIns = bytearray(0x0C.to_bytes(1, byteorder='big'))
    if opcodes.is_push(op):
        size = opcodes.operand_size(op)
        for i in range(size):
            retIns += bytearray(0x0C.to_bytes(1, byteorder='big'))
    return retIns


def getpushid(jumpdest):
    pushid = 0x60
    size = len(jumpdest) * 4 // 8
    pushid += size - 1
    return pushid, size


def getopsize(instr):
    op = instr.op
    size = 1
    if opcodes.is_push(op):
        size += opcodes.operand_size(op)
    return size


def convert(contr, dfg, patches, bytecode, retbasic, endpc):
    """
    Convert patched contract to bytecode
    """

    instrs = contr.instructions
    basics = retbasic.values()
    append_basics = []
    addbytecode = bytearray()

    for pc, instr in instrs.items():
        # Add patching instructions
        for b in basics:
            if instr in b.instructions:
                #print(hex(pc), hex(b.end))
                if pc == b.offset:
                    jumpdest = hex(endpc + len(addbytecode))
                    print(jumpdest)
                    #print(jumpdest)
                    pushid, size = getpushid(jumpdest)
                    jumpdest = int(jumpdest, 16)
                    #生成trampoline的基本块
                    bytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                    bytecode += bytearray(jumpdest.to_bytes(size, byteorder='big'))
                    bytecode += bytearray(0x56.to_bytes(1, byteorder='big'))
                    addbytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))
                    size += 2
                    #print(size)


                # #push不可能是基本块末尾
                # if pc == b.end:
                #     bytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))

                if size == 0:
                    if pc == b.end:
                        if contr.instructions[pc + 1].name != "JUMPDEST":
                            bytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))
                        else:
                            bytecode += instruction_to_bytecodeINVALID(instr)
                    else:
                        bytecode += instruction_to_bytecodeINVALID(instr)
                else:
                    opsize = getopsize(instr)
                    if opsize <= size:
                        size -= opsize
                    else:
                        opsize -= size
                        size = 0
                        for i in range(opsize):
                            bytecode += bytearray(0x0C.to_bytes(1, byteorder='big'))

                if pc in patches:
                    for _, slices in sorted(patches[pc].items()):
                        for off in slices:
                            addbytecode += instruction_to_bytecode(instrs[off])

                # Remove discarded instructions
                if instr.discarded:
                    # 丢弃的指令则不增加bytecode,并且
                    for pre in dfg.predecessors(pc):
                        # Replace discarded instructions with POPs
                        if not instrs[pre].discarded:
                            # 0x50 pop
                            # 当该指令被舍弃而前一个指令未被舍弃时，因为前置关系，故需要将其从栈中pop出来
                            addbytecode += bytearray(0x50.to_bytes(1, byteorder='big'))
                else:
                    addbytecode += instruction_to_bytecode(instr)

                if pc == b.end:
                    if contr.instructions[pc + 1].name != "JUMPDEST":
                        jumpdest = hex(pc)
                    else:
                        jumpdest = hex(pc + 1)
                    pushid, size = getpushid(jumpdest)
                    jumpdest = int(jumpdest, 16)
                    addbytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                    addbytecode += bytearray(jumpdest.to_bytes(size, byteorder='big'))
                    addbytecode += bytearray(0x56.to_bytes(1, byteorder='big'))
                break
        else:
            bytecode += instruction_to_bytecode(instr)
    bytecode += addbytecode


def convert_mode2(contr, call, callbasic, bytecode, endpc, free_storage):
    """
    Convert patched contract to bytecode
    """

    instrs = contr.instructions
    addbytecode = bytearray()

    for pc, instr in instrs.items():
        # Add patching instructions
        if instr in callbasic.instructions:
            if pc == callbasic.offset:
                jumpdest = hex(endpc + len(addbytecode))
                # print(jumpdest)
                pushid, size = getpushid(jumpdest)
                jumpdest = int(jumpdest, 16)
                # 生成trampoline的基本块
                bytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                bytecode += bytearray(jumpdest.to_bytes(size, byteorder='big'))
                bytecode += bytearray(0x56.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))
                size += 2
                # print(size)

            # #push不可能是基本块末尾
            # if pc == b.end:
            #     bytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))

            if size == 0:
                if pc == callbasic.end:
                    if contr.instructions[pc + 1].name != "JUMPDEST":
                        bytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))
                else:
                    bytecode += instruction_to_bytecodeINVALID(instr)
            else:
                opsize = getopsize(instr)
                if opsize <= size:
                    size -= opsize
                else:
                    opsize -= size
                    size = 0
                    for i in range(opsize):
                        bytecode += bytearray(0x0C.to_bytes(1, byteorder='big'))
            if pc == call:
                pushid, size = getpushid(hex(free_storage))
                addbytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(free_storage.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(0x54.to_bytes(1, byteorder='big'))
                #iszero
                addbytecode += bytearray(0x15.to_bytes(1, byteorder='big'))
                #计算jumpdest—call_loc
                jumpdest = endpc + len(addbytecode)
                pushid, size = getpushid(hex(jumpdest))
                jumpdest += 3 + size
                pushid, size = getpushid(hex(jumpdest))
                addbytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(jumpdest.to_bytes(size, byteorder='big'))
                #jumpi
                addbytecode += bytearray(0x57.to_bytes(1, byteorder='big'))
                #revert
                addbytecode += bytearray(0xFD.to_bytes(1, byteorder='big'))
                #jumpdest
                addbytecode += bytearray(0x5B.to_bytes(1, byteorder='big'))
                #push 1
                addbytecode += bytearray(0x60.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(0x1.to_bytes(1, byteorder='big'))
                #push free_storage
                pushid, size = getpushid(hex(free_storage))
                addbytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(free_storage.to_bytes(1, byteorder='big'))
                #storage[free_storage] = 1
                addbytecode += bytearray(0x55.to_bytes(1, byteorder='big'))
                #call
                addbytecode += bytearray(0xF1.to_bytes(1, byteorder='big'))
                #push 0
                addbytecode += bytearray(0x60.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(0x0.to_bytes(1, byteorder='big'))
                #push free_storage
                pushid, size = getpushid(hex(free_storage))
                addbytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(free_storage.to_bytes(1, byteorder='big'))
                #storage[free_storage] = 0
                addbytecode += bytearray(0x55.to_bytes(1, byteorder='big'))
            else:
                addbytecode += instruction_to_bytecode(instr)

            if pc == callbasic.end:
                if contr.instructions[pc + 1].name != "JUMPDEST":
                    jumpdest = hex(pc)
                else:
                    jumpdest = hex(pc + 1)
                pushid, size = getpushid(jumpdest)
                jumpdest = int(jumpdest, 16)
                addbytecode += bytearray(pushid.to_bytes(1, byteorder='big'))
                addbytecode += bytearray(jumpdest.to_bytes(size, byteorder='big'))
                addbytecode += bytearray(0x56.to_bytes(1, byteorder='big'))

        else:
            bytecode += instruction_to_bytecode(instr)
    bytecode += addbytecode



def getbytecode(contr, call, callbasic, dfg, patches, retbasic, endpc, mode, free_storage):
    """
    Restore patched contract to bytecode
    """

    bytecode = bytearray()
    print(mode)
    if mode == 0:
        convert(contr, dfg, patches, bytecode, retbasic, endpc)
    else:
        convert_mode2(contr, call, callbasic, bytecode, endpc, free_storage)
    return bytecode