import json


def SBT(cur_root_id, node_list):
    cur_root = node_list[cur_root_id]
    tmp_list = []
    tmp_list.append("(")
    text_list = []
    text_list.append("(")
    if 'value' in cur_root and cur_root['value'] != 'None':
        value = cur_root['value']
        # str = cur_root['type'] + '_' + value
        str = cur_root['type']
    else:
        value = 'None'
        str = cur_root['type']
    text_list.append(value)
    tmp_list.append(str)

    if 'children' in cur_root:
        chs = cur_root['children']
        for ch in chs:
            tmpl, textl = SBT(ch, node_list)
            tmp_list.extend(tmpl)
            text_list.extend(textl)

    tmp_list.append(")")
    tmp_list.append(str)

    text_list.append(")")
    text_list.append(value)

    return tmp_list, text_list


def get_sbt(asts):
    ast_sbt, text = SBT(0, asts)
    sbt = ' '.join(ast_sbt)
    return sbt, text
