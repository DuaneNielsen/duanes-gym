import torch

height = 10
width = 10
layers = 3


def int2tensor(x, y):
    if isinstance(x, int) and isinstance(y, int):
        return torch.tensor([x]), torch.tensor([y])
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) > 0 and len(y) > 0:
            if isinstance(x[0], int) and isinstance(y[0], int):
                return torch.tensor(x), torch.tensor(y)
    elif isinstance(x, (list, tuple)) and y is None:
        if len(x) > 0 and isinstance(x[0], tuple):
            xl = []
            yl = []
            for xc, yc in x:
                xl.append(xc)
                yl.append(yc)
            return torch.tensor(xl), torch.tensor(yl)
    else:
        return None, None


class Vec:
    def __init__(self, x, y=None):
        self.x, self.y = int2tensor(x, y)
        if isinstance(x, str):
            xl = []
            yl = []
            for d in x:
                xl.append(direction[d].x)
                yl.append(direction[d].y)
            self.x = torch.tensor(xl)
            self.y = torch.tensor(yl)
        if self.x is None or self.y is None:
            raise Exception


class Pos:
    def __init__(self, x, y=None):
        self.x, self.y = int2tensor(x, y)
        if self.x is None or self.y is None:
            raise Exception

    def __add__(self, other):
        if isinstance(other, Vec):
            self.x += other.x
            self.y += other.y
            return self
        else:
            raise Exception

    def __eq__(self, other):
        if isinstance(other, Pos):
            return torch.all(self.x == other.x) and torch.all(self.y == other.y)
        if isinstance(other, tuple):
            return torch.all(self.x == other[0]) and torch.all(self.y == other[1])


color = {
    'red':          0b1,
    'orange':      0b10,
    'yellow':     0b100,
    'green':     0b1000,
    'blue':     0b10000,
    'indigo':  0b100000,
    'violet': 0b1000000
}

north = Vec(0, 1)
east = Vec(1, 0)
south = Vec(0, -1)
west = Vec(-1, 0)

direction = {
    'N': Vec(0, 1),
    'E': Vec(1, 0),
    'S': Vec(0, -1),
    'W': Vec(-1, 0)
}

action_map = [
    Vec(0, 1),
    Vec(1, 0),
    Vec(0, -1),
    Vec(-1, 0)
]


class Grid:
    def __init__(self, width, height, instances=1):
        self.static = torch.tensor((instances, width, height, layers), dtype=torch.half)
        self.dynamic = torch.tensor((instances, width, height, layers), dtype=torch.half)

    def add(self, entity):
        pass


class Entity:
    def __init__(self, pos):
        self.pos = pos
        self.space = None

    def addToSpace(self, space):
        raise NotImplementedError


class Player(Entity):
    def __init__(self, pos, color='red'):
        super().__init__(pos)
        self.color = color[color]
        self.color_t = torch.tensor()

    def move(self, direction):
        self.space.dynamic[:, self.pos.x, self.pos.y] = self.space.dynamic[:, self.pos.x, self.pos.y] | self.color
        self.pos += direction
        self.space.dynamic[:, self.pos.x, self.pos.y] = self.space.dynamic[:, self.pos.x, self.pos.y] | self.color

    def addToSpace(self, space):
        self.space = space
        self.space.dynamic[:, self.pos.x, self.pos.y] = self.space.dynamic[:, self.pos.x, self.pos.y] | self.color

