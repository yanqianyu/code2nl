import javalang
import json
import collections


def get_ast(text):
        code = ""
        code = text.replace('\n', ' ')
        # for line in text:
        #    code += line
        # print(code)
    # for line in code:
        # ----

        # ----
        tokens = javalang.tokenizer.tokenize(code)
        token_list = list(javalang.tokenizer.tokenize(code))

        # ----
        for i in range(len(token_list)):
            if type(token_list[i]) is javalang.tokenizer.String \
                       or type(token_list[i]) is javalang.tokenizer.Character:
                token_list[i].value = '<STR>'
            if type(token_list[i]) is javalang.tokenizer.Integer \
                    or type(token_list[i]) is javalang.tokenizer.DecimalInteger \
                    or type(token_list[i]) is javalang.tokenizer.OctalInteger \
                    or type(token_list[i]) is javalang.tokenizer.BinaryInteger \
                    or type(token_list[i]) is javalang.tokenizer.HexInteger \
                    or type(token_list[i]) is javalang.tokenizer.FloatingPoint \
                    or type(token_list[i]) is javalang.tokenizer.DecimalFloatingPoint \
                    or type(token_list[i]) is javalang.tokenizer.HexFloatingPoint:
                token_list[i].value = '<NUM>'
        tokens = (i for i in token_list)
        # ----

        length = len(token_list)
        parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse_member_declaration()
        except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
            print(code)
            return None
            # continue
        flatten = []
        for path, node in tree:
            flatten.append({'path': path, 'node': node})

        ign = False
        outputs = []
        stop = False
        for i, Node in enumerate(flatten):
            d = collections.OrderedDict()
            path = Node['path']
            node = Node['node']
            children = []
            for child in node.children:
                child_path = None
                if isinstance(child, javalang.ast.Node):
                    child_path = path + tuple((node,))
                    for j in range(i + 1, len(flatten)):
                        if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                            children.append(j)
                if isinstance(child, list) and child:
                    child_path = path + (node, child)
                    for j in range(i + 1, len(flatten)):
                        if child_path == flatten[j]['path']:
                            children.append(j)
            d["id"] = i
            d["type"] = str(node)
            if children:
                d["children"] = children
            value = None
            if hasattr(node, 'name'):
                value = node.name
            elif hasattr(node, 'value'):
                value = node.value
            elif hasattr(node, 'position') and node.position:
                for i, token in enumerate(token_list):
                    if node.position == token.position:
                        pos = i + 1
                        value = str(token.value)
                        while (pos < length and token_list[pos].value == '.'):
                            value = value + '.' + token_list[pos + 1].value
                            pos += 2
                        break
            elif type(node) is javalang.tree.This \
                    or type(node) is javalang.tree.ExplicitConstructorInvocation:
                value = 'this'
            elif type(node) is javalang.tree.BreakStatement:
                value = 'break'
            elif type(node) is javalang.tree.ContinueStatement:
                value = 'continue'
            elif type(node) is javalang.tree.TypeArgument:
                value = str(node.pattern_type)
            elif type(node) is javalang.tree.SuperMethodInvocation \
                    or type(node) is javalang.tree.SuperMemberReference:
                value = 'super.' + str(node.member)
            elif type(node) is javalang.tree.Statement \
                    or type(node) is javalang.tree.BlockStatement \
                    or type(node) is javalang.tree.ForControl \
                    or type(node) is javalang.tree.ArrayInitializer \
                    or type(node) is javalang.tree.SwitchStatementCase:
                value = 'None'
            elif type(node) is javalang.tree.VoidClassReference:
                value = 'void.class'
            elif type(node) is javalang.tree.SuperConstructorInvocation:
                value = 'super'

            if value is not None and type(value) is type('str'):
                d['value'] = value
            if not children and not value:
                # print('Leaf has no value!')
                print(type(node))
                ign = True

                # break
            outputs.append(d)

        # outputs = json.dump(outputs)

        return outputs
