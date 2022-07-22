import time
from copy import deepcopy
from collections import deque, defaultdict

import numpy as np
from hexalattice import hexalattice
from matplotlib import pyplot as plt
from termcolor import cprint
from test_cases import lattice0, lattice1, lattice2, lattice3, lattice4, lattice5, lattice6, lattice7

GUESSES = 0
CALLS = 0


class Hidato:
    def __init__(self, lattice, domains=None, assignment=None, initial_cells=None, anchors=None, distance=None):
        self.lattice = deepcopy(lattice)
        self.height = len(lattice)
        self.min_width = len(lattice[0])
        self.max_value = sum(self.min_width + i for i in range(self.height // 2 + 1)) * 2 - len(
            self.lattice[self.height // 2])
        self.variables = [i for i in range(1, self.max_value+1)]
        self.cells = {(i, j) for i in range(self.height) for j in range(len(self.lattice[i]))}

        if initial_cells is None:
            self.initial_cells = {cell for cell in self.cells if self.lattice[cell[0]][cell[1]] != 0}
        else:
            self.initial_cells = initial_cells.copy()

        if assignment is None:
            self.assignment = {}
            for i, j in self.cells:
                if self.lattice[i][j] != 0:
                    self.assignment[self.lattice[i][j]] = (i, j)
        else:
            self.assignment = assignment.copy()

        if anchors is None:
            self.anchors = self.get_anchors()
        else:
            self.anchors = deepcopy(anchors)

        if distance is None:
            self.distance = self.get_distances_from_anchors()
        else:
            self.distance = deepcopy(distance)

        if domains is None:
            self.domains = {n: (self.cells - set(self.assignment.values())).copy()
                            if n not in self.assignment else {self.assignment[n]}
                            for n in self.variables}
            forward_check(self)
        else:
            self.domains = deepcopy(domains)

    def assign(self, n, cell, get_anchors=False, get_distance_from_anchors=False):
        """Assign value cell to variable n and consequently:
            1) Update the lattice at cell to be n
            2) Add n to assignment
            3) Update the domain of n to be {cell} and remove cell from the domains of all other variables
            4) Update the anchors
            5) Update distance to anchors"""

        self.lattice[cell[0]][cell[1]] = n
        self.assignment[n] = cell
        self.domains[n] = {cell}
        for var in self.domains:
            if var != n and cell in self.domains[var]:
                self.domains[var].remove(cell)
        if get_anchors:
            self.anchors = self.get_anchors()
        if get_distance_from_anchors:
            self.distance = self.get_distances_from_anchors()

    def get_anchors(self):
        """return a dictionary that for every unassigned number keeps the anchors of that number. the anchors of
        of a number are the closet assigned numbers from below and from above"""
        anchors = dict()
        curr_anchor = None
        for n in range(1, self.max_value+1):
            if n not in self.assignment:
                anchors[n] = [curr_anchor]
            else:
                curr_anchor = n
        curr_anchor = None
        for n in range(self.max_value, 0, -1):
            if n not in self.assignment:
                anchors[n].append(curr_anchor)
            else:
                curr_anchor = n
        return anchors

    def get_distances_from_anchors(self):
        """Returns a dictionary that maps every anchor to another dictionary of all reachable cells from anchor
        and the distance from anchors to the cell"""

        distance = dict()
        all_anchors = set([anchor for anchors_pair in self.anchors.values() for anchor in anchors_pair
                           if anchor is not None])
        for anchor in all_anchors:
            distance[anchor] = self.get_distances_from_cell(self.assignment[anchor])

        return distance

    def get_distances_from_cell(self, s):
        """return a dictionary mapping every cell reachable from s using only empty cells to the distance to that cell
        from s"""

        distance = {s: 0}
        queue = deque([s])

        while queue:
            v = queue.popleft()
            for u in self.neighbors(v):
                if u not in distance and self.lattice[u[0]][u[1]] == 0:
                    distance[u] = distance[v] + 1
                    queue.append(u)
        del distance[s]
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

    def plot(self, initial_cells_only=False):
        lattice_flattened = [((i, j), n) for i, row in enumerate(self.lattice) for j, n in enumerate(row)]
        num_cells = len(lattice_flattened)
        min_width = len(self.lattice[0])
        r = min_width - 1
        colors = np.ones((num_cells, 3))
        for i in range(num_cells):
            if lattice_flattened[i][0] in self.initial_cells:
                colors[i] = [193 / 255, 252 / 255, 239 / 255]
        hex_centers, _ = hexalattice.create_hex_grid(nx=3 * r,
                                                     ny=3 * r,
                                                     crop_circ=r,
                                                     face_color=colors[::-1],
                                                     do_plot=True)
        hex_centers = hex_centers[::-1]
        for i, hex_center in enumerate(hex_centers):
            if (lattice_flattened[i][1] != 0
                    and (not initial_cells_only or
                         (initial_cells_only and lattice_flattened[i][0] in self.initial_cells))):
                plt.text(hex_center[0], hex_center[1], lattice_flattened[i][1], fontsize=15, ha='center', va='center')
        plt.show()

    def neighbors(self, cell):
        """Returns a list containing all neighbors of cell"""
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

    def copy(self):
        return Hidato(self.lattice, self.domains, self.assignment, self.initial_cells, self.anchors, self.distance)


def solved(hidato):
    """Check whether the hidato is solved. Defining N as the number of cells in the hidato lattice then A Hidato is
     solved if all cells are filled with numbers from 1 to N and there is a connected path through neighbors path from 1
     to N"""
    return (all(n in hidato.assignment for n in hidato.variables)
            and all(hidato.assignment[n] in hidato.neighbors(hidato.assignment[n - 1])
                    for n in range(2, hidato.max_value + 1)))


def select_unassigned_variable(hidato):
    """Return an unassigned variable using the Most Restricted Variable heuristic"""
    return min([var for var in hidato.variables
                if var not in hidato.assignment],
               key=lambda var: len(hidato.domains[var]))


def order_domain_values(var, hidato):
    """Return the domain of var"""
    return hidato.domains[var]


def possible(cell, n, hidato):
    """Check if it is possible to assign n to cell. That is, the cell is empty and  if n-1 or n+1 are assigned,
    they must be in cells adjacent to cell"""
    if hidato.lattice[cell[0]][cell[1]] != 0:
        return False

    if n-1 in hidato.assignment and hidato.assignment[n-1] not in hidato.neighbors(cell):
        return False

    if n+1 in hidato.assignment and hidato.assignment[n+1] not in hidato.neighbors(cell):
        return False
    return True


def reduce_domains(hidato):
    """
    Go over all unassigned variables. for every unassigned variable m, go over all its possible cells in domain
    and keep in domain only cells where, for each anchor of n, the cell is reachable from the anchor and
    0 < distance[cell_anchor][cell_n] <= abs(anchor-n). If upon reducing the domains a variable is left with an
    empty domain, then it means there is a contradiction. If a variable is left with a single value in its domain
    then, assuming there is no contradiction, this is a new inference that means we can later assign the variable to the
    this value.

    Arguments:
    hidato - an instance of class Hidato representing a hidato puzzle

    Returns:
    contradiction - boolean indicating whether hidato contains a contradiction, i.e, it is unsolvable
    new_inferences - dictionary of n:cell pairs meaning that assuming contradiction is false then
                     value cell can be assigned to variable n for every n:cell pair in new_inferences.
    """
    contradiction = False
    new_inferences = dict()
    new_cells = set()
    for m in hidato.variables:
        if m not in hidato.assignment:
            hidato.domains[m] = set(filter(lambda c: all([c in hidato.distance[a]
                                                          and 0 < hidato.distance[a][c] <= abs(a - m)
                                                          for a in hidato.anchors[m] if a is not None]),
                                           hidato.domains[m]))
            if len(hidato.domains[m]) == 1:
                m_cell = list(hidato.domains[m])[0]
                if m_cell in new_cells:
                    contradiction = True
                    break
                new_inferences[m] = m_cell
                new_cells.add(m_cell)
            if hidato.domains[m] == set():
                contradiction = True
                break
    return contradiction, new_inferences


def find_naked_cells(hidato):
    """Make new inferences by looking at the domains and finding cells which only appear in a domain of a single
    variable n.

    Arguments:
    hidato - an instance of class Hidato representing a hidato puzzle

    Returns:
    contradiction - boolean indicating whether hidato contains a contradiction, i.e, it is unsolvable
    new_inferences - dictionary of n:cell pairs meaning that assuming contradiction is false then
                     value cell can be assigned to variable n for every n:cell pair in new_inferences.
    """

    empty_cells = hidato.cells - set(hidato.assignment.values())
    cell_domains = defaultdict(set)
    new_inferences = dict()

    for cell in empty_cells:
        for n, domain in hidato.domains.items():
            if cell in domain:
                cell_domains[cell].add(n)

    for cell, cell_domain in cell_domains.items():
        if cell_domain == set():
            return True, new_inferences
        elif len(cell_domain) == 1:
            n = list(cell_domain)[0]
            new_inferences[n] = cell

    return False, new_inferences


def forward_check(hidato):
    """
    Make inferences from current state of hidato and return whether a contradiction was found when making the
    inferences.

    Arguments:
    hidato - an instance of class Hidato representing a hidato puzzle

    Returns:
    contradiction - boolean indicating whether hidato contains a contradiction, i.e, it is unsolvable
        """
    while True:
        contradiction_1, new_inferences_1 = reduce_domains(hidato)
        contradiction_2, new_inferences_2 = find_naked_cells(hidato)
        contradiction = contradiction_1 or contradiction_2
        new_inferences = {**new_inferences_1, **new_inferences_2}
        if contradiction:
            return contradiction
        elif not new_inferences:
            return False
        else:
            for var, val in new_inferences.items():
                hidato.assign(var, val)
            hidato.anchors = hidato.get_anchors()
            hidato.distance = hidato.get_distances_from_anchors()


def solve(hidato):
    global GUESSES
    global CALLS
    CALLS += 1
    if solved(hidato):
        print("solved!")
        return hidato
    var = select_unassigned_variable(hidato)
    if len(hidato.domains[var]) > 1:
        GUESSES += 1
    for val in order_domain_values(var, hidato):
        if possible(val, var, hidato):
            new_hidato = hidato.copy()
            new_hidato.assign(var, val, get_anchors=True, get_distance_from_anchors=True)
            contradiction = forward_check(new_hidato)
            if contradiction:
                continue
            result = solve(new_hidato)
            if result is not None:
                return result
    return None


# TODO - maybe incorporate naked pairs/triples/quadruplets etc.. instead of just naked singles.

puzzle = Hidato(lattice5)
puzzle.plot(initial_cells_only=True)
t0 = time.perf_counter()
solution = solve(puzzle)
t1 = time.perf_counter()
print(f"Solved the hidato In {t1-t0} seconds using {GUESSES} guesses and {CALLS} calls")
solution.plot()
