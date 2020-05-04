import sys
import numpy as np
import heapq
from collections import namedtuple
import time


class puzzle_state:
    previous_state = None
    state = None
    moves = 0
    cost = 0
    space_index = -1
    direction = ""
    kind_heuristic = 0

    # constructor for the puzzle state
    def __init__(self, state, previous_state=None, moves=0, direction="", kind_of_heuristic=0):
        self.previous_state = previous_state
        self.state = np.array(state)
        self.moves = moves
        self.direction = direction
        self.kind_heuristic = kind_of_heuristic

    def check_for_goal(self):   # check if current state has reached goal
        goal_state = np.arange(9)
        if np.array_equal(self.state, goal_state):
            return True
        return False

    def __lt__(self, other_state):  # function to sort heap according to cost (heauristic function)
        if (self.cost != other_state.cost):
            return self.cost < other_state.cost


class movement:
    visited = None

    def __init__(self, visited):
        self.visited = visited
        self.space_index = self.find_space_index()

    def find_space_index(self):     # find the place of the empty tile
        for i in range(9):
            if (self.visited.state[i] == 0):
                return i
        return -1
    # function that returns a new object with the given elements switch places

    def swap_places(self, element1, element2):
        temp_state = np.array(self.visited.state)
        temp_state[element1], temp_state[element2] = temp_state[element2], temp_state[element1]
        return temp_state

    def move_up(self):
        # valid to move up
        if(self.space_index > 2):
            return puzzle_state(self.swap_places(self.space_index, self.space_index - 3), self.visited, self.visited.moves + 1, "UP", self.visited.kind_heuristic)
        else:
            return None

    def move_down(self):
        # valid to move down
        if(self.space_index < 6):
            return puzzle_state(self.swap_places(self.space_index, self.space_index + 3), self.visited, self.visited.moves + 1, "DOWN", self.visited.kind_heuristic)
        else:
            return None

    def move_left(self):
        # valid to move left
        if(self.space_index % 3 != 0):
            return puzzle_state(self.swap_places(self.space_index, self.space_index - 1), self.visited, self.visited.moves + 1, "LEFT", self.visited.kind_heuristic)
        else:
            return None

    def move_right(self):
        # valid to move right
        if(self.space_index % 3 != 2):
            return puzzle_state(self.swap_places(self.space_index, self.space_index + 1), self.visited, self.visited.moves + 1, "RIGHT", self.visited.kind_heuristic)
        else:
            return None

    def find_neighbours(self):  # generate all possible neighbours for a state
        neighbours = []
        neighbours.append(self.move_up())
        neighbours.append(self.move_down())
        neighbours.append(self.move_left())
        neighbours.append(self.move_right())
        return list(filter(None, neighbours))   # remove none from the list of neighbours


def createid(x):
    return str(x)


class puzzle_solver:    # class that is responsible for solving any puzzle by any chosen mean
    solved_puzzle = None
    total_cost = 0
    nodes_expanded = 0

    def dfs_search(self, puzzle):
        frontierqueue = {}
        explored = {}
        frontierqueue.update({createid(puzzle.state): puzzle})
        while(len(frontierqueue) != 0):
            visited = frontierqueue.popitem()[1]
            explored.update({createid(visited.state): visited})
            if visited.check_for_goal():
                self.solved_puzzle = visited
                self.nodes_expanded = len(explored)-1
                return
            neigbours = movement(visited).find_neighbours()
            for neighbour in reversed(neigbours):
                if ((explored.get(createid(neighbour.state)) == None) and ((frontierqueue.get(createid(neighbour.state)) == None))):
                    frontierqueue.update({createid(neighbour.state): neighbour})

    def bfs_search(self, puzzle):
        frontierqueue = {}
        explored = {}
        frontierorg = list()
        frontierqueue.update({createid(puzzle.state): puzzle})
        frontierorg.append(puzzle)
        while(len(frontierqueue) != 0):
            visited = frontierorg.pop(0)
            frontierqueue.pop(str(visited.state))
            explored.update({createid(visited.state): visited})
            if visited.check_for_goal():
                self.solved_puzzle = visited
                self.nodes_expanded = len(frontierorg)-1
                return
            neigbours = movement(visited).find_neighbours()
            for neighbour in (neigbours):
                if ((explored.get(createid(neighbour.state)) == None) and ((frontierqueue.get(createid(neighbour.state)) == None))):
                    frontierqueue.update({createid(neighbour.state): neighbour})
                    frontierorg.append(neighbour)

    def A_star_search(self, puzzle):
        # main idea is to put the initial state into a minimume heap that returns the puzzle state with the smallest cost among the neighbours
        neighbours_heap = []
        explored_states_list = []
        if puzzle.kind_heuristic == 2:
            puzzle.cost = puzzle.moves + self.euclidean_cost(puzzle)
        else:
            puzzle.cost = puzzle.moves + self.manhatten_cost(puzzle)
        heapq.heappush(neighbours_heap, puzzle)
        while(len(neighbours_heap) != 0):
            least_cost_state = heapq.heappop(neighbours_heap)
            explored_states_list.append(least_cost_state)
            if least_cost_state.check_for_goal():
                self.solved_puzzle = least_cost_state
                self.nodes_expanded = len(explored_states_list)-1
                return
            for neighbour in movement(least_cost_state).find_neighbours():
                if not any(np.array_equal(neighbour.state, i.state) for i in explored_states_list):
                    if neighbour.kind_heuristic == 2:
                        neighbour.cost = neighbour.moves + self.euclidean_cost(neighbour)
                    else:
                        neighbour.cost = neighbour.moves + self.manhatten_cost(neighbour)
                    heapq.heappush(neighbours_heap, neighbour)

    def manhatten_cost(self, puzzle):
        state = puzzle.state
        goal = np.arange(9)
        return np.sum((np.absolute(state // 3 - goal // 3) + np.absolute(state % 3 - goal % 3)))

    def euclidean_cost(self, puzzle):
        state = puzzle.state
        goal = np.arange(9)
        return np.sum(np.sqrt((np.square(state // 3 - goal // 3) + np.sqrt(np.square(state % 3 - goal % 3)))))

    def get_best_path(self):
        path_list_array = []
        path_list_objects = []
        current_state = self.solved_puzzle
        path_list_objects.append(current_state)
        path_list_array.append(current_state.state)
        while current_state.previous_state != None:
            current_state = current_state.previous_state
            path_list_objects.append(current_state)
            path_list_array.append(current_state.state)
        path_list_array.reverse()
        path_list_objects.reverse()
        return path_list_array, path_list_objects

    def get_goal_directions(self):
        b, path_obj_list = self.get_best_path()
        direction_list = []
        for item in path_obj_list:
            if item.direction != "":
                direction_list.append(item.direction)
        return direction_list

    def get_total_cost(self):
        _, path_list_obj = self.get_best_path()
        return len(path_list_obj)-1


def main():
    choice = 0
    heauristic_choice = 0
    initial_state = []
    print("*" * 50, end="")
    print("welcome to 8-puzzle game", end="")
    print("*" * 50)
    for i in range(0, 9):
        initial_state.append(0)
    while True:
        print("please enter initial state ::")
        for i in range(0, 9):
            initial_state[i] = int(input())
        # printing the initial state
        print("take a look at your entered state in a clearer representation >>")
        print()
        for i in range(9):
            print(initial_state[i], end=" ")
            if (i + 1) % 3 == 0:
                print()
        initial_state = np.array(initial_state)
        puzzle_helper = puzzle_solver()
        print("*" * 50)
        print("select the algorithm to solve the puzzle ::")
        print("1- BFS")
        print("2- DFS")
        print("3- A*")
        print("4- exit")
        choice = int(input())
        puzzle = puzzle_state(initial_state, None, 0, "", 0)
        print("*" * 50)
        if choice == 1:
            print("you chose BFS algorithm")
            t0 = time.time()
            puzzle_helper.bfs_search(puzzle)
            t1 = time.time()
        elif choice == 2:
            print("you chose DFS algorithm")
            t2 = time.time()
            puzzle_helper.dfs_search(puzzle)
            t3 = time.time()
        elif choice == 3:
            print("you chose A* algorithm")
            print("please choose the heauristic function to be used ::")
            print("1- manhatten")
            print("2- euclidean")
            heauristic_choice = int(input())
            t4 = time.time()
            if heauristic_choice == 1:
                puzzle = puzzle_state(initial_state, None, 0, "", 1)
                puzzle_helper.A_star_search(puzzle)
            elif heauristic_choice == 2:
                puzzle = puzzle_state(initial_state, None, 0, "", 2)
                puzzle_helper.A_star_search(puzzle)
            else:
                while heauristic_choice != 1 and heauristic_choice != 2:
                    heauristic_choice = int(input("wrong choice try again :: "))
            t5 = time.time()
        elif choice == 4:
            break
        else:
            while choice != 1 and choice != 2 and choice != 3:
                choice = int(input("wrong choice try again :: "))
        print("solving the puzzle please wait....")
        path, n = puzzle_helper.get_best_path()
        print("path taken to reach goal >> ", path)
        print("direction taken to reach goal >> ", puzzle_helper.get_goal_directions())
        print("goal reached >> ", puzzle_helper.solved_puzzle.state)
        print("total cost to reach goal is ", puzzle_helper.get_total_cost())
        print("nodes expanded >> ", puzzle_helper.nodes_expanded)
        print("depth is ", puzzle_helper.solved_puzzle.moves)
        if choice == 1:
            print("running time of BFS is >> {} secs".format(t1-t0))
        elif choice == 2:
            print("running time of DFS is >> {} secs".format(t3-t2))
        elif choice == 3:
            print("running time of A* is >> {} secs".format(t5-t4))
        print("*" * 50)


if __name__ == "__main__":
    main()
