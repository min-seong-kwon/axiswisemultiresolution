
###
octree_orders = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
    ]

###
class TreeNode(object):
    def is_branch(self):
        return (type(self) == BranchNode)

    def is_leaf(self):
        return type(self) == LeafNode

###
class BranchNode(TreeNode):
    def __init__(self, parent=None):
        self.parent = parent
        self.child = []
   
###
class LeafNode(TreeNode):
    def __init__(self, vertex=[0,0,0], value=0.0, weight=0.0, R=0.0, G=0.0, B=0.0, parent=None):
        self.parent = parent
        self.vertex = vertex
        self.value = value
        self.weight = weight
        self.R = R
        self.G = G
        self.B = B