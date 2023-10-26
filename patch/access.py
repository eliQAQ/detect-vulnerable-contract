import logging
import networkx as nx
import opcodes
import abstract
import hierarchy

log = logging.getLogger(__name__)


class Contract:
    def __init__(self):
        self.blocks = []
        self.instructions = {}
        self.jump_destination = {}


class BasicBlock:
    def __init__(self, offset):
        self.offset = offset
        self.end = None
        self.instructions = []
        self.next = None
        self.revert = False
        self.layer = None
    def setEnd(self, end):
        self.end = end

class Instruction:
    def __init__(self, op, name, arg):
        self.op = op
        self.name = name
        self.arg = arg
        self.reserved = False
        self.layer = None
        self.dependence = set()
        self.overwrite = set()
        self.discarded = False


def initialize(evm):
    contr = Contract()
    dfg = nx.DiGraph()
    cfg = nx.DiGraph()
    cur_blk = BasicBlock(0)
    pc = 0
    while pc < len(evm):
        op = evm[pc]
        if op not in opcodes.listing:
            raise KeyError('Invalid op. op: {:#x}, offset: {:#x}'.format(op, pc))

        name = opcodes.listing[op][0]
        size = opcodes.operand_size(op)
        if size != 0:
            arg = int.from_bytes(evm[pc + 1:pc + 1 + size], byteorder='big')
        else:
            arg = None

        instr = Instruction(op, name, arg)
        if name == 'JUMPDEST':
            if len(cur_blk.instructions) > 0:
                contr.blocks.append(cur_blk)  #将当前块加入合约记录的块当中
                cfg.add_node(cur_blk.offset, blk=cur_blk) #控制流图
                #以JUMPDEST为新的基本块
                cur_blk.setEnd(pc - 1)
                new_blk = BasicBlock(pc)
                cur_blk.next = new_blk
                cur_blk = new_blk
            cur_blk.offset += 1
            #记录合约的跳转
            contr.jump_destination[pc] = cur_blk
            contr.instructions[pc] = instr
        else:
            cur_blk.instructions.append(instr)
            contr.instructions[pc] = instr

            if opcodes.is_swap(op) or opcodes.is_dup(op):
                pass
            elif (name == 'JUMP' or name == 'JUMPI' or name == 'STOP' or name == 'RETURN' or
                  name == 'REVERT' or name == 'INVALID' or name == 'SUICIDE' or name == 'CALL'):

                if name == 'CALL':
                    if len(cur_blk.instructions) <= 4:
                        dfg.add_node(pc, instr=instr)
                        pc += size + 1
                        continue

                contr.blocks.append(cur_blk)
                cur_blk.setEnd(pc)
                cfg.add_node(cur_blk.offset, blk=cur_blk)
                new_blk = BasicBlock(pc + 1)
                #这里的next存储的是顺序链接
                cur_blk.next = new_blk
                cur_blk = new_blk
                dfg.add_node(pc, instr=instr)
            else:
                dfg.add_node(pc, instr=instr)

        pc += size + 1

    #判断是否最后仍有一个基本块没记录，或者只有jumpdest，也算一个基本块
    if len(cur_blk.instructions) > 0 or cur_blk.offset - 1 in contr.jump_destination:
        contr.blocks.append(cur_blk)
        cfg.add_node(cur_blk.offset, blk=cur_blk)
    else:
        contr.blocks[-1].next = None

    return contr, dfg, cfg, pc

def analyze(contr, dfg, cfg):
    trace = {}
    free_storage = [1]
    # 获取到dfg, cfg, trace
    # trace来源于CALL和SSTORE之间的序列
    abstract.execute(contr, dfg, cfg, trace, free_storage)
    hierarchy.layering(contr, cfg)
    return trace, free_storage
