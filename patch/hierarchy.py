import logging

log = logging.getLogger(__name__)


class StackElement:
    def __init__(self, block, new, previous_layer):
        self.block = block
        self.new = new
        self.previous_layer = previous_layer


def set_layers(blk, layer):
    blk.layer = layer
    for instr in blk.instructions:
        instr.layer = layer


def layering(contr, cfg):
    # 总的来说就两种划分方式入度大于1以及会revert的块都另起一层
    # 虽然我不太认可这种作法
    cnt = 0
    init_blk = contr.blocks[0].offset
    trv_stack = [StackElement(init_blk, True, None)]
    nodes = cfg.nodes()

    while len(trv_stack) > 0:
        cur_elm = trv_stack.pop(0)
        blk = cur_elm.block

        if nodes[blk]['blk'].layer is not None:
            continue
        else:
            # 当它有多个入度时，则新生成一层
            if cfg.in_degree(blk) > 1:
                layer = cnt
                cnt += 1
            else:
                # 如果这个块之前未遇见过，则新生成一层
                if cur_elm.new:
                    layer = cnt
                    cnt += 1
                else:
                    layer = cur_elm.previous_layer

            # 把block中的每一条指令都设立layer
            set_layers(nodes[blk]['blk'], layer)

        hold_layer = False
        non_revert = None
        for next_blk in cfg[blk]:
            trv_stack.insert(0, StackElement(next_blk, True, None))
            # 判断是否需要revert
            if not nodes[next_blk]['blk'].revert:
                # 如果该块不是需要revert的块，则把该块标记为non_revert
                # 如果是，则
                if non_revert is None:
                    non_revert = trv_stack[0]
                    hold_layer = True
                else:
                    hold_layer = False
        if hold_layer:
            # 该块非第一次遍历到
            non_revert.new = False
            non_revert.previous_layer = layer
