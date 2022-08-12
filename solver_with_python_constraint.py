import time
from copy import deepcopy
from collections import deque
from constraint import Problem, AllDifferentConstraint
from termcolor import cprint
from test_cases import lattices


class Hidato:
    def __init__(self, lattice):
        self.lattice = lattice
        self.height = len(lattice)
        self.min_width = len(lattice[0])
        self.max_value = sum(self.min_width + i for i in range(self.height // 2 + 1)) * 2 - len(
            self.lattice[self.height // 2])
        self.assignment = dict()
        self.cells = set()
        self.initial_cells = set()
        self.distance = dict()
        for i, row in enumerate(lattice):
            for j, value in enumerate(row):
                self.cells.add((i, j))
                self.distance[(i, j)] = self.get_distances_from_cell((i, j))
                if value != 0:
                    self.initial_cells.add((i, j))
                    self.assignment[value] = (i, j)

    def get_distances_from_cell(self, s):
        distance = {s: 0}
        queue = deque([s])

        while queue:
            v = queue.popleft()
            for u in self.neighbors(v):
                if u not in distance:
                    distance[u] = distance[v] + 1
                    queue.append(u)

        return distance

    def print_cell(self, cell, end='\n'):
        on_color = 'on_blue' if cell in self.initial_cells else 'on_white'
        value = self.lattice[cell[0]][cell[1]]
        value_str = '  ' if value == 0 else (value if len(str(value)) == 2 else ' ' + str(value))
        cprint(value_str, on_color=on_color, end=end)

    def print(self):
        n_spaces = self.height // 2
        for i, row in enumerate(self.lattice):
            print(n_spaces * '  ', end='')
            for j, value in enumerate(row):
                if j != len(row) - 1:
                    self.print_cell((i, j), end='  ')  # print(cell if len(str(cell))==2 else f'0{cell}',end='  ')
                else:
                    self.print_cell((i, j))  # print(cell if len(str(cell))==2 else f'0{cell}')
            if i < self.height // 2:
                n_spaces -= 1
            else:
                n_spaces += 1

    def neighbors(self, cell):
        i, j = cell
        if (not (0 <= i < self.height)) or (not (0 <= j < len(self.lattice[i]))):
            raise ValueError(f" cell {cell} does not exist")

        ngbs = []
        if j > 0:
            ngbs.append((i, j - 1))
        if j < len(self.lattice[i]) - 1:
            ngbs.append((i, j + 1))

        if i > 0:  # Add neighbors from row above
            if i <= self.height // 2:
                if j - 1 >= 0:
                    ngbs.append((i - 1, j - 1))
                if j < len(self.lattice[i - 1]):
                    ngbs.append((i - 1, j))
            else:
                ngbs.append((i - 1, j))
                ngbs.append((i - 1, j + 1))

        if i < self.height - 1:  # Add neighbors from row below
            if i < self.height // 2:
                ngbs.append((i + 1, j))
                ngbs.append((i + 1, j + 1))
            else:
                if j - 1 >= 0:
                    ngbs.append((i + 1, j - 1))
                if j < len(self.lattice[i + 1]):
                    ngbs.append((i + 1, j))

        return ngbs


def set_variables_and_domains(hidato):
    problem = Problem()
    for n in range(1, hidato.max_value+1):
        if n in hidato.assignment:
            problem.addVariable(n, [hidato.assignment[n]])
        else:
            problem.addVariable(n, list(hidato.cells-set(hidato.assignment.values())))
    return problem


def add_hidato_constraints(problem, hidato):
    for n in range(2, hidato.max_value+1):
        problem.addConstraint(constraint=lambda x, y: x in hidato.neighbors(y),
                              variables=(n-1, n))
    problem.addConstraint(AllDifferentConstraint())


def generate_distance_constraint(d, hidato):
    def f(x, y):
        return 0 <= hidato.distance[x][y] <= d
    return f


def add_hidato_constraints_v2(problem, hidato):
    for n1 in range(1, hidato.max_value):
        for n2 in range(n1+1, hidato.max_value+1):
            problem.addConstraint(constraint=generate_distance_constraint(d=n2-n1, hidato=hidato),
                                  variables=(n1, n2))
    problem.addConstraint(AllDifferentConstraint())


def get_lattice(hidato, hidato_assignment):
    new_lattice = deepcopy(hidato.lattice)
    hidato_assignment_reversed = {v: k for k, v in hidato_assignment.items()}
    for i, row in enumerate(hidato.lattice):
        for j, cell in enumerate(row):
            new_lattice[i][j] = hidato_assignment_reversed[(i, j)]
    return new_lattice


def solve_with_csp(hidato):
    problem = set_variables_and_domains(hidato)
    # add_hidato_constraints(problem, hidato)
    add_hidato_constraints_v2(problem, hidato)
    solution = Hidato(get_lattice(hidato, problem.getSolution()))
    solution.initial_cells = hidato.initial_cells.copy()
    return solution


print("Problem:")
puzzle = Hidato(lattices['l5'])
puzzle.print()
print("Solution:")
t0 = time.perf_counter()
puzzle_solution = solve_with_csp(puzzle)
t1 = time.perf_counter()
print(t1-t0)
puzzle_solution.print()
