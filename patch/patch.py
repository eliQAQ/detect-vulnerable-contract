import logging

log = logging.getLogger(__name__)


def slicing(source, call, dfg, sequence, discarded):
    # call是call的指令
    # source是当前的指令
    slices = []
    deps = set()
    overs = set()
    trv_stack = [source]
    visited = {node: False for node in dfg.nodes()}
    nodes = dfg.nodes()
    # 意思是sstore肯定要丢弃
    discarded.add(source)

    while len(trv_stack) > 0:
        instr = trv_stack[0]
        if visited[instr]:
            visited[instr] = False
            # 后序遍历后，又先序输出
            slices.append(instr)
            trv_stack.pop(0)
        else:
            visited[instr] = True
            if instr == call:
                raise RuntimeError('Violating data flow dependencies. CALL: {:#x}, lift: {:#x}'.format(call, instr))
            if not nodes[instr]['instr'].reserved:
                # 枚举数据流的相关指令
                # suc是后继啦
                for suc in dfg[instr]:
                    if suc not in discarded:
                        break
                # 循环正常进行则将该指令加入丢弃序列，即后继所有指令都需丢弃
                # 如果它后继的所有指令都需要丢弃，则它也需要丢弃
                else:
                    discarded.add(instr)

            # dependence则记录的是和地址相关的东西
            # overwrite也是记录的修改地址的东西
            for dep in nodes[instr]['instr'].dependence:
                if dep in sequence:
                    # 如果dependence里面还包括该序列则说明
                    deps.add(dep)
            overs.update(nodes[instr]['instr'].overwrite)
            for pre in dfg.predecessors(instr):
                # 后序遍历，重复说明有环，报错
                if visited[pre]:
                    raise RuntimeError('Error slicing, loop detected. lift: {:#x}, pre: {:#x}'.format(instr, pre))
                trv_stack.insert(0, pre)
    return slices, deps, overs


def lifting(sstore, call, dfg, trace, sliced, lifted, discarded):
    # 判断call,sstore是否在记录序列中
    if (call, sstore) not in trace:
        raise KeyError('Error tracing executed instructions from CALL to SSTORE. CALL: {:#x}, SSTORE: {:#x}'
                       .format(call, sstore))
    # 初始栈只放sstore的指令位置
    lift_stack = [sstore]
    # 获取记录的call,sstore trace集合
    sequence = trace[(call, sstore)]
    # 获取数据流中的节点
    nodes = dfg.nodes()
    #
    lifts = set()
    overwrites = set()
    dependencies = set()
    num = 0
    while len(lift_stack) > 0:
        num += 1
        if num >= 10000:
            raise RuntimeError('loop')
        instr = lift_stack.pop(0)
        # if nodes[call]['instr'].layer != nodes[instr]['instr'].layer:
        #     raise RuntimeError('Violating control flow dependencies. CALL: {:#x}, lift: {:#x}'.format(call, instr))
        name = nodes[instr]['instr'].name
        num += 1

        # 如果中间的指令遇到这几个调用，则报错
        if name == 'CALL' or name == 'CALLCODE' or name == 'DELEGATECALL' or name == 'STATICCALL':
            raise RuntimeError('Cannot lift CALL, CALLCODE, DELEGATECALL or STATICCALL. CALL: {:#x}, lift: {:#x}'
                               .format(call, instr))

        # 开始分片
        slices, deps, overs = slicing(instr, call, dfg, sequence, discarded)
        # slices指从当前指令开始前序找所有与之关联的指令
        lifts.update(slices)
        overwrites.update(overs)
        if instr not in sliced:
            sliced[instr] = slices
        if instr not in lifted:
            lifted[instr] = {call: sequence}
        else:
            if call not in lifted[instr]:
                for pos in dict(lifted[instr]):
                    if call in lifted[instr][pos]:
                        break
                    elif pos in sequence:
                        lifted[instr][call] = sequence
                        del lifted[instr][pos]
                        break
                else:
                    lifted[instr][call] = sequence

        for dep in deps:
            lift_stack.insert(0, dep)
    for instr in sequence.difference(lifts):
        dependencies.update(nodes[instr]['instr'].dependence)
    # 重写的指令里面不能有dependencies相关的指令
    if len(overwrites.intersection(dependencies)) > 0:
        raise RuntimeError('Violating memory/storage dependencies. CALL: {:#x}, SSTORE: {:#x}'.format(call, sstore))


def set_report(report, call, sstore, msg):
    report['Reentrancy'].append(
        {
            'callOffset': call,
            'sStoreOffset': sstore,
            'result': msg
        }
    )


def execute(dfg, trace, reentrancy, report):
    sliced = {}
    lifted = {}
    discarded = set()
    for vul in reentrancy:
        call = vul[0]
        sstore = vul[1]
        old_sliced = sliced.copy()
        old_lifted = lifted.copy()
        old_discarded = discarded.copy()
        try:
            lifting(sstore, call, dfg, trace, sliced, lifted, discarded)
        except Exception as e:
            if str(e).strip('\'') == 'Timeout.':
                raise e
            else:
                sliced = old_sliced
                lifted = old_lifted
                discarded = old_discarded
                set_report(report, call, sstore, str(e).strip('\''))
        else:
            set_report(report, call, sstore, 'Done.')
    nodes = dfg.nodes()
    for instr in discarded:
        nodes[instr]['instr'].discarded = True
    patches = {}
    for instr in lifted:
        for pos in lifted[instr]:
            if pos not in patches:
                patches[pos] = {}
            # 但其实所有lifted里面都是 call:sequence啊
            # 也就是存储的patches[call][instr]
            patches[pos][instr] = sliced[instr]
    return call, patches
