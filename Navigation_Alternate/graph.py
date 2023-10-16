"""
Author: Thomas Pardy
Date Modified: <2023-08-21 Mon>
Module containing graph related classes for mapping of arena.
"""
import math
import heapq


class Node:
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.distance = float("inf")
        self.neighbours = []
        self.prev_node = None
        self.xy = [float("inf"), float("inf")]
        self.is_obstacle = False
        self.is_aruco = False
        self.is_target = False
        self.is_fruit = False
        self.fruit_name = None
        self.aruco_num = -1
        self.ghf = [0,0,0]

    def add_neighbour(self, neighbour, weight):
        self.neighbours.append((neighbour, weight))

    def __lt__(self, other_node):
        return self.distance < other_node.distance


class Graph:
    """
    Contains graph module, with shortest path methods.
    """

    def __init__(self):
        self.nodes = {}

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    def __getitem__(self, pos) -> Node:
        """get_node: Returns node given name of node"""
        i, j = pos
        return self.nodes[f"({i},{j})"]

    def og_get(self, ind):
        return self.nodes[ind]

    def distance(self, pos1, pos2) -> float:
        """distance: Helper function, returns distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def distance_x(self, x1, x2):
        return abs(x1 - x2)

    def distance_x(self, x1, x2):
        return abs(x1 - x2)

    def reset_graph(self):
        for node in self.nodes:
            self.nodes[node].visited = False
            self.nodes[node].prev_node = None
            self.nodes[node].distance = float('inf')

    def get_nearest_node(self, pos):
        """get_nearest_node: Takes in a (x,y) position, returns nearest node on map"""
        min_dist = float("inf")
        min_node = None
        x1, y1 = pos
        for node_name in self.nodes:
            node = self.nodes[node_name]
            dist = self.distance((x1, y1), node.xy)
            if dist < min_dist and node.is_obstacle == False:
                min_dist = dist
                min_node = node

        return min_node

    def adjacent_nodes(self, node, object_size, circle_flag) -> list:
        """adjacent_nodes: returns surrounding nodes within a radius"""
        def recursive_nodes(current_node, object_size, pos, memo, circle_flag):
            x_radius, y_radius = object_size[0] / 2, object_size[1] / 2
            memo.append(current_node)
            
            for neighbour, _ in current_node.neighbours:
                if neighbour not in memo:
                    if circle_flag and self.distance(neighbour.xy, pos) < x_radius:
                        recursive_nodes(neighbour, object_size, pos, memo, circle_flag)
                    elif not circle_flag:
                        # Optionally, if not using a circular boundary, check both x and y distances
                        x_dist = abs(neighbour.xy[0] - pos[0])
                        y_dist = abs(neighbour.xy[1] - pos[1])
                        if x_dist < x_radius and y_dist < y_radius:
                            recursive_nodes(neighbour, object_size, pos, memo, circle_flag)

        memo = []
        recursive_nodes(node, object_size, node.xy, memo, circle_flag)
        return memo


    def set_obstacle(self, node) -> None:
        """set_obstacle: Given a node sets it as an obstacle in the graph"""
        node.is_obstacle = True
        for neighbour, _ in node.neighbours:
            neighbour.neighbours = [x for x in neighbour.neighbours if x[0] != node]
            # ?
        node.neighbours = []

    def djikstras(self, start_node, target_node) -> None:
        """djikstras: Returns shortest path between two nodes"""
        start_node.distance = 0
        heap = [(0, start_node)]

        while heap:
            current_distance, current_node = heapq.heappop(heap)

            if current_node == target_node:
                break

            if current_node.visited:
                continue

            current_node.visited = True

            for neighbour,  weight in current_node.neighbours:
                if not neighbour.visited and not neighbour.is_obstacle:
                    new_distance = current_distance + weight
                    if new_distance < neighbour.distance:
                        neighbour.distance = new_distance
                        neighbour.prev_node = current_node
                        heapq.heappush(heap, (new_distance, neighbour))
    def a_star(self, start_node, end_node):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        start_node.ghf = [0,0,0]
        end_node.ghf = [0,0,0]

        open_list = []
        closed_list = []

        open_list.append(start_node)

        while len(open_list) > 0:
            current_node = open_list[0]
            current_index = 0

            for index, item in enumerate(open_list):
                if item.ghf[2] < current_node.ghf[2]:
                    current_node = item
                    current_index = index

            open_list.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.name)
                    current = current.prev_node
                return path[::-1]
            
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                node_position = (current_node.xy[0] + new_position[0], current_node.xy[1] + new_position[1])

                if node_position[0] > (len(self.nodes) - 1) or node_position[0] < 0 or node_position[1] > (len(self.nodes.og_get(len(self.nodes)-1) -1)) or node_position[1] < 0:
                    continue

                new_node = self.nodes[f"({node_position[0]},{node_position[1]})"]
                if new_node.is_obstacle:
                    continue

                children.append(new_node)

            for child in children:
                if child in closed_list:
                    continue

                child.ghf[0] = current_node.ghf[0] + 1
                child.ghf[1] = self.distance(child.xy, end_node.xy)
                child.ghf[2] = child.ghf[0] + child.ghf[1]

                for open_node in open_list:
                    if child == open_node and child.ghf[0] > open_node.ghf[0]:
                        continue

                open_list.append(child)

    def get_shortest_distance(self, target: Node):
        """get_shortest_distance: Returns shortest distance and path to target node"""
        path = []
        current_node = target

        while current_node.prev_node is not None:
            path.insert(0, current_node.name)
            current_node = current_node.prev_node
        path.insert(0, current_node.name)
        return target.distance, path


if __name__ == "__main__":
    nodes = [[Node(f"({row},{col})") for col in range(3)] for row in range(3)]

    for row in range(3):
        for col in range(3):
            if row > 0:
                nodes[row][col].add_neighbour(nodes[row - 1][col], 1)  # Up

            if row < 2:
                nodes[row][col].add_neighbour(nodes[row + 1][col], 1)  # Down

            if col > 0:
                nodes[row][col].add_neighbour(nodes[row][col - 1], 1)  # Left

            if col < 2:
                nodes[row][col].add_neighbour(nodes[row][col + 1], 1)  # Right

    G = Graph()
    for row in nodes:
        for node_i in row:
            node_i.xy = [row.index(node_i), nodes.index(row)]
            G.add_node(node_i)

    start_node = nodes[2][2]
    target_node = nodes[0][1]
    G.a_star(start_node, target_node)
