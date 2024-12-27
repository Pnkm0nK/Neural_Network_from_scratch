# class for further computational graph work and forward-,backward propagations
from graphviz import Digraph
import math


class Value:
    def __init__(self, value, children=(), operation='', label='', grad=0):
        self.value = value
        self.label = label
        self._children = set(children)
        self._op = operation
        self._backward = lambda: None
        self._grad = grad

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        # to add constants we wrap it in value class
        other = other if isinstance(other, Value) else Value(other)

        result = Value(self.value + other.value, (self, other), '+')

        def backward():
            self._grad += result._grad
            other._grad += result._grad

        result._backward = backward
        return result

    def __radd__(self, other):
        # to add const + Value class
        return self + other

    def __sub__(self, other):
        # to add constants we wrap it in value class
        other = other if isinstance(other, Value) else Value(other)

        result = self + (-other)
        return result

    def __rsub__(self, other):
        # to add const + Value class
        return self - other

    def __mul__(self, other):
        # to multiply constants we wrap it in value class
        other = other if isinstance(other, Value) else Value(other)

        result = Value(self.value * other.value, (self, other), '*')

        def backward():
            self._grad += other.value * result._grad
            other._grad += self.value * result._grad

        result._backward = backward
        return result

    def __rmul__(self, other):
        # to multiply Value on a constant in reversed order e.g. 3 * Value(2.0)
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = self * (other**(-1))
        return result

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        result = Value(self.value ** power, children=(self,), operation=f"^{power}")

        def backward():
            self._grad += power * result.value / self.value * result._grad

        result._backward = backward
        return result

    def exp(self):
        result = Value(math.exp(self.value), children=(self,), operation='exp')

        def backward():
            self._grad += result.value * result._grad

        result._backward = backward
        return result

    def relu(self):
        result = Value(max(self.value, 0), children=(self,), operation="relu")

        def _backward():
            self._grad += 1 if self.value > 0 else 0

        result._backward = _backward
        return result

    def tanh(self):
        exponent = (2*self).exp()
        out = (exponent - 1) / (exponent + 1)
        return out

    def backprop(self):
        def topo_sort(root):
            # topological sort(all edges from left to right) for computational graph
            # to call _backward func in correct order
            visited = set()
            topo = []

            def topo_traversal(node):
                if node not in visited:
                    visited.add(node)
                    for child in node._children:
                        topo_traversal(child)
                    topo.append(node)

            topo_traversal(root)
            return topo

        sorted_comp_graph = topo_sort(self)
        # traverse in reverse order and call _backward on all values
        # because it is ordered from the last child to parents
        # and we need to start from parents to children
        for i in range(len(sorted_comp_graph)-1, -1, -1):
            sorted_comp_graph[i]._backward()



def get_nodes_and_edges(root):
    nodes = set()
    edges = set()

    def DFS(nd):
        nodes.add(nd)
        for child in nd._children:
            edges.add((child, nd))
            DFS(child)

    DFS(root)
    return nodes, edges


def draw_comp_graph(root):
    # Digraph from graph vizualization that is displayed from left to right
    comp_graph = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = get_nodes_and_edges(root)
    for nd in nodes:
        uid = str(id(nd))
        # for any graph value create rect node
        comp_graph.node(name=uid, label=f"{nd.label} | value:{nd.value:.3f} | grad: {nd._grad:.3f}", shape='record')
        if nd._op:
            # if it is an operation create an operation node
            comp_graph.node(name=uid+nd._op, label=nd._op)
            comp_graph.edge(uid+nd._op, uid)

    for nd1, nd2 in edges:
        # create edges between value nodes
        comp_graph.edge(str(id(nd1)), str(id(nd2))+nd2._op)
    return comp_graph


if __name__ == "__main__":
    # test for backprop
    x1 = Value(2)
    x1.label = 'x1'
    w1 = Value(-3)
    w1.label = 'w1'
    x2 = Value(1)
    x2.label = 'x2'
    w2 = Value(0)
    w2.label = 'w2'
    x1w1 = x1 * w1
    x1w1.label = 'x1w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2w2'
    b = Value(4, label='b')
    sumx1w1x2w2 = x1w1+x2w2
    sumx1w1x2w2.label = "x1w1+x2w2"
    n = x1w1+x2w2 + b
    # ----tanh-----
    e = (2*n).exp()
    out = (e - 1) / (e + 1)
    out.label = 'tanh'
    out._grad = 1.0
    out.backprop()

    graph = draw_comp_graph(out)
    output_file = "graph_visualize"
    graph.render(output_file, format='pdf', cleanup=True)

