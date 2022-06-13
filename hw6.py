
class Neuron:

    def __init__(self, w, f = lambda x: x):
        self.w_list = w
        self.func_act = f

    def forward(self, x):
        self.x_list = x
        s = sum([x+w for x, w in zip(x, self.w_list)])
        return self.func_act(s)

    def backlog(self):
        if 'x_list' in self.__dict__.keys():
            return self.x_list
        else:
            return

n = Neuron([1, 2, 3, 4, 5])
print(n.backlog())
print(n.forward([10, 20, 30, 40, 50]))
print(n.backlog())

