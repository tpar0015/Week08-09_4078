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

    def __init__(self, arena_dimensions):
        self.nodes = {}
        self.arena_dimensions = arena_dimensions

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

    def get_nearest_node(self, pos, alter_node=(0,0)):
        """get_nearest_node: Takes in a (x,y) position, returns nearest node on map"""
        min_dist = float("inf")
        min_nodes = []
        x1, y1 = pos
        for node_name in self.nodes:
            node = self.nodes[node_name]
            dist = self.distance((x1, y1), node.xy)
            if dist < min_dist and node.is_obstacle == False:
                min_dist = dist
                min_nodes = [node]
            elif dist == min_dist and node.is_obstacle == False:
                min_nodes.append(node)
        min_dist_to_center = float('inf')
        for node in min_nodes:
            if self.distance(node.xy, alter_node) < min_dist_to_center:
                min_dist_to_center = self.distance(node.xy, alter_node)
                min_node = node

        if len(min_nodes) > 1:
            print("Min Nodes:", [n.xy for n in min_nodes])
            print(alter_node)
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

                current_node_name = eval(current_node.name)
                node_name = (current_node_name[0] + new_position[0], current_node_name[1] + new_position[1])
                node_position = self.get_nearest_node(node_name).xy

                if node_position[0] > self.arena_dimensions[0]/2 or node_position[0] < -self.arena_dimensions[0]/2 or node_position[1] > self.arena_dimensions[1]/2 or node_position[1] < -self.arena_dimensions[1]/2:
                    continue

                new_node = self.nodes[f"({node_name[0]},{node_name[1]})"]
                if new_node.is_obstacle:
                    print('node is obstacle')
                    continue

                children.append(new_node)

            for child in children:
                if child in closed_list:
                    continue

                child.ghf[0] = current_node.ghf[0] + 1
                child.ghf[1] = self.distance(child.xy, end_node.xy)
                child.ghf[2] = child.ghf[0] + child.ghf[1]

                child.prev_node = current_node

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
    arena_dimensions = (1000, 100)
    radius = 10
    G = Graph(arena_dimensions)
    # Computes and initializes number of nodes
    row_n = arena_dimensions[0] // radius - 1
    col_n = arena_dimensions[1] // radius - 1
    nodes = []
    for i in range(row_n):
        row = []
        for j in range(col_n):
            node = Node(f"({i},{j})")
            # Center is 0,0
            node.xy = [i*radius + radius - arena_dimensions[0]/2, j*radius + radius - arena_dimensions[1]/2]
            #node.xy = [i*self.radius + self.radius, j*self.radius + self.radius]

            row.append(node)

        nodes.append(row)


    # Adds neighbours to corresponding nodes
    for i in range(row_n):
        for j in range(col_n):
            if i > 0:
                # nodes[i][j].add_neighbour(nodes[i - 1][j], self.G.distance(nodes[i][j].xy, nodes[i-1][j].xy)) # Up
                nodes[i][j].add_neighbour(nodes[i - 1][j], 1) # Up
            if i < row_n - 1:
                # nodes[i][j].add_neighbour(nodes[i + 1][j], self.G.distance(nodes[i][j].xy, nodes[i+1][j].xy))  # Down
                nodes[i][j].add_neighbour(nodes[i + 1][j], 1)  # Down
            if j > 0:
                # nodes[i][j].add_neighbour(nodes[i][j - 1], self.G.distance(nodes[i][j].xy, nodes[i][j-1].xy)) # Left
                nodes[i][j].add_neighbour(nodes[i][j - 1], 1) # Left
            if j < col_n - 1:
                # nodes[i][j].add_neighbour(nodes[i][j + 1], self.G.distance(nodes[i][j].xy, nodes[i][j+1].xy)) # Right
                nodes[i][j].add_neighbour(nodes[i][j + 1], 1)
    # Adds nodes to graph
    for i in nodes:
        for j in i:
            G.add_node(j)
    start_node = G.get_nearest_node((0,0))
    G.set_obstacle(G.get_nearest_node((250,0)))
    end_node = G.get_nearest_node((500,0))
    print([print(G[eval(node)].xy) for node in G.a_star(start_node, end_node)])

