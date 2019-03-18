class BinaryTreeNode(object):
    def __init__(self, val=None, type=None, lch=None, rch=None):
        self.val = val
        self.lch = lch
        self.rch = rch
        self.type = type


def preorder_traverse(cur_root):
    if cur_root is None:
        return []
    lst = []
    if cur_root.val and cur_root.val != 'None':
        lst.append(cur_root.type + '_' + cur_root.val)
    else:
        lst.append(cur_root.type)
    lst.extend(preorder_traverse(cur_root.lch))
    lst.extend(preorder_traverse(cur_root.rch))
    return lst


def preorder_structure(cur_root):
    if cur_root is None:
        return []
    lst = []
    lst.append(cur_root.type)
    lst.extend(preorder_structure(cur_root.lch))
    lst.extend(preorder_structure(cur_root.rch))
    return lst


def inorder_traverse(cur_root):
    if cur_root is None:
        return []
    lst = []
    lst.extend(inorder_traverse(cur_root.lch))
    if cur_root.val and cur_root.val != 'None':
        lst.append(cur_root.type + '_' + cur_root.val)
    else:
        lst.append(cur_root.type)
    lst.extend(inorder_traverse(cur_root.rch))
    return lst


def postorder_traverse(cur_root):
    if cur_root is None:
        return []
    lst = []
    lst.extend(postorder_traverse(cur_root.lch))
    lst.extend(postorder_traverse(cur_root.rch))
    if cur_root.val and cur_root.val != 'None':
        lst.append(cur_root.type + '_' + cur_root.val)
    else:
        lst.append(cur_root.type)
    return lst


def show(cur_root, depth=0):
    print("  " * depth, end='')
    print(cur_root.val)
    if cur_root.lch is not None and cur_root.rch is not None:
        show(cur_root.lch, depth + 1)
        show(cur_root.rch, depth + 1)
    elif cur_root.lch is None and cur_root.rch is not None:
        print("  " * (depth + 1) + "*")
        show(cur_root.rch, depth + 1)
    elif cur_root.lch is not None and cur_root.rch is None:
        show(cur_root.lch, depth + 1)
        print("  " * (depth + 1) + "*")
    else:
        return
