"""
 -
Author: Thomas Pardy
Date Modified: 2023-08-23

"""
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/tom/ECE4078/ECE4078_Lab_2023/Week08-09_4078/')
from Navigation_Alternate.graph import Graph, Node
from math import comb
import w8HelperFunc as w8
from PIL import Image
# Recieve position from rotary encoders
# Recieve readouts from ultrasonic sensor

class Map:
    """Generates map o arena, navigates shortest path. Online updating of path
    with detected obstacles factored in"""
    def __init__(self, arena: tuple, radius: float, true_map: str, shopping_list: str, 
                 aruco_size = (300,300), fruit_size = (300,300), target_size=(200,200), distance_threshold=200,
                 plot=True):
        """
        Initializes variables
        """
        self.arena_dimensions = arena
        self.distance_threshold = distance_threshold
        self.radius = radius
        self.center_obstacles = []
        self.obstacle_radius = []
        self.G = Graph(arena)
        self.location = (0,0,0)
        self.path = []
        self.object_size = (250,90)
        self.obstacle_corners = []
        self.true_map = true_map
        self.shopping_list = shopping_list
        self.aruco_size = aruco_size
        self.fruit_size = fruit_size
        self.target_size = target_size
        self.circle_flag = False
        self.plot = plot
        print(plot)

    def generate_map(self):
        """
        Generates grid-like nodal map of arena.
        """
        # Computes and initializes number of nodes
        row_n = self.arena_dimensions[0] // self.radius - 1
        col_n = self.arena_dimensions[1] // self.radius - 1
        nodes = []
        for i in range(row_n):
            row = []
            for j in range(col_n):
                node = Node(f"({i},{j})")
                # Center is 0,0
                node.xy = [i*self.radius + self.radius - self.arena_dimensions[0]/2, j*self.radius + self.radius - self.arena_dimensions[1]/2]
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
                self.G.add_node(j)

    def update_location(self, pose) -> None:
        """
        Updates predicted location on nodal map
        """
        self.location = pose


    def add_obstacles(self, obs_xy, object_size: tuple, is_fruit=False, is_target=False, is_aruco=False) -> None:
        """
        Re calibrates map with object blocked out on nodes.
        """
        obs_x, obs_y = obs_xy
        # self.G.reset_graph()
        closest_node = self.G.get_nearest_node((obs_x, obs_y))
        obstacle_nodes = self.G.adjacent_nodes(closest_node, object_size, self.circle_flag)
        obstacle_xy = []
        for node in obstacle_nodes:
            obstacle_xy.append(node.xy)
        corners = [(min([x[0] for x in obstacle_xy]) - 1*self.radius, min([x[1] for x in obstacle_xy]) - 1*self.radius),
                    (min([x[0] for x in obstacle_xy]) - 1*self.radius, max([x[1] for x in obstacle_xy]) + 1*self.radius),
                    (max([x[0] for x in obstacle_xy]) + 1*self.radius, min([x[1] for x in obstacle_xy]) - 1*self.radius),
                    (max([x[0] for x in obstacle_xy]) + 1*self.radius, max([x[1] for x in obstacle_xy]) + 1*self.radius),
                    ]
        self.obstacle_corners.append(corners)

        for node in obstacle_nodes:
            if is_fruit:
                node.is_fruit = True
            elif is_aruco:
                node.is_aruco = True
            elif is_target:
                node.is_target = True
            self.G.set_obstacle(node)

    def add_aruco_markers(self):
        """
        Adds aruco markers to map
        """
        _, _, aruco_positions = w8.read_true_map(self.true_map)
        i = 1
        for aruco in aruco_positions:
            aruco = aruco * 1000
            self.G.get_nearest_node(aruco).aruco_num = i
            self.add_obstacles(aruco, self.aruco_size, is_aruco=True)
            i += 1

    def add_fruits_as_obstacles(self):
        fruit_targets = w8.read_search_list(self.shopping_list)
        fruit_list, fruit_pos, _ = w8.read_true_map(self.true_map)
        fruit_coords = w8.read_target_fruits_pos(fruit_targets, fruit_list, fruit_pos)
        for fruit, name in zip(fruit_pos, fruit_list):
            fruit = fruit * 1000
            if name in fruit_targets:
                self.G.get_nearest_node(fruit).fruit_name = name
                self.add_obstacles(fruit, (self.target_size),is_fruit=True, is_target=True)
            else:
                self.G.get_nearest_node(fruit).fruit_name = name
                self.add_obstacles(fruit, (self.fruit_size), is_fruit=True)



    def update_path(self, start_node, waypoint) -> None:
        """
        Updates path to avoid any new obstacles
        """
        end_node = self.G.get_nearest_node(waypoint[:2])
        end_node.is_target = True
        if end_node is not None:
            self.G.djikstras(start_node, end_node)
            _, path = self.G.get_shortest_distance(end_node)
            # path = self.G.a_star(start_node, end_node)
            path.insert(0,start_node.name)
            self.path.append(path)
        else:
            print("Cant find waypoint.")
        self.path = self.shorten_shortest_path()

    def get_path_xy(self) -> list:
        """
        Returns path
        """
        full_path_xy = []
        for i in range(len(self.path)):
            path = self.path[i]
            path_xy = []
            for node in path:
                path_xy.append(self.G[eval(node)].xy)
            full_path_xy.append(path_xy)
        return full_path_xy

    def check_obstacle(self, ultrasonic_readout: float, detect_distance) -> bool:
        """Checks if ultrasonic readout detected obstacle, if so remaps and updates path"""

        if ultrasonic_readout < detect_distance:
            self.remap(ultrasonic_readout, self.object_size)
            self.update_path
            return True
        else:
            return False
    def ccw(self, A, B, C):
        """ccw: Returns true if points are in counter clockwise order"""
        val = (C[1]-A[1]) * (B[0]-A[0]) - (B[1]-A[1]) * (C[0]-A[0])
        if val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0
        
    def line_intersect(self, A, B, C, D):
        """line_intersect: Returns true if line segments AB and CD intersect"""
        # return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
        o1 = self.ccw(A, C, D)
        o2 = self.ccw(B, C, D)
        o3 = self.ccw(A, B, C)
        o4 = self.ccw(A, B, D)
        if ((o1 != o2) and (o3 != o4)):
            return True
        return False

    def line_obstacle_free_square(self, A, B):
        """
        line_obstacle_free: Returns true if line segment AB is obstacle free
        """
        for corner in self.obstacle_corners:
            if self.line_intersect(A.xy, B.xy, corner[0], corner[1]) or self.line_intersect(A.xy, B.xy, corner[0], corner[2]) or self.line_intersect(A.xy, B.xy, corner[1], corner[3]) or self.line_intersect(A.xy, B.xy, corner[2], corner[3]):
                return False
        return True
    
    def line_obstacle_free_circle(self, A, B):
            # y = mx + c
            if B[0] - A[0] == 0:
                a = 1
                b = 0
                c = -A[0]
            else:
                m = (B[1] - A[1]) / (B[0] - A[0])
                a = -m
                b = 1
                c = m * A[0] - A[1]
            for i in range(len(self.center_obstacles)):
                x,y = self.center_obstacles[i]
                dist = ((abs(a * x + b * y + c)) / math.sqrt(a * a + b * b))
                if dist + self.radius <= self.obstacle_radius[i]:
                    return False


    def shorten_shortest_path(self) -> None:
        """
        Shortens path by removing nodes that are not needed.
        """
        threshold_distance = self.distance_threshold
        final_path = []
        for path in self.path:

            start_node = self.G[eval(path[0])]
            i = 1
            while i < len(path) - 1:
                current_node = self.G[eval(path[i])]
                
                # Check if line between start and current node is obstacle free
                if self.circle_flag:
                    if self.line_obstacle_free_square(start_node, current_node) and self.G.distance(start_node.xy, current_node.xy) < threshold_distance:
                        path.pop(i)

                    else:
                        start_node = current_node
                        i += 1
                else:
                    if self.line_obstacle_free_square(start_node, current_node) and self.G.distance(start_node.xy, current_node.xy) < threshold_distance:
                        path.pop(i)

                    else:
                        start_node = current_node
                        i += 1
            final_path.append(path)
        return final_path
    
    def get_targets(self):
        fruit_targets = w8.read_search_list(self.shopping_list)
        fruit_list, fruit_pos, _ = w8.read_true_map(self.true_map)
        fruit_coords = w8.read_target_fruits_pos(fruit_targets, fruit_list, fruit_pos)
        for i in range(len(fruit_coords)):
            x, y = fruit_coords[i] 
            target_fruit = (x*1000, y*1000)
            if i == 0:
                self.update_path(self.G.get_nearest_node(self.location[:2]), target_fruit)
            else:
                self.update_path(self.G[eval(self.path[-1][-1])], target_fruit)
            self.G.reset_graph()

    def draw_arena_v2(self):
        G_img = nx.Graph()
        node_attributes = {}
        edge_attributes = {}
        max_x = self.arena_dimensions[0]/2
        max_y = self.arena_dimensions[1]/2
        G_img.add_node("A")
        node_attributes["A"] = {"pos": (-max_x, -max_y), "size": 10, "color": "black"}
        G_img.add_node("B")
        node_attributes["B"] = {"pos": (max_x, -max_y), "size": 10, "color": "black"}
        G_img.add_node("C")
        node_attributes["C"] = {"pos": (max_x, max_y), "size": 10, "color": "black"}
        G_img.add_node("D")
        node_attributes["D"] = {"pos": (-max_x, max_y), "size": 10, "color": "black"}
        G_img.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        edge_attributes[("A", "B")] = {"color": "black", "width": 1}
        edge_attributes[("B", "C")] = {"color": "black", "width": 1}
        edge_attributes[("C", "D")] = {"color": "black", "width": 1}
        edge_attributes[("D", "A")] = {"color": "black", "width": 1}

        flattened_path = [[0,0]]
        for path in self.path:
            for node in path:
                flattened_path.append(node)

        # Remove duplicates that are adjacent
        flattened_path = [flattened_path[i] for i in range(1,len(flattened_path))if flattened_path[i] != flattened_path[i-1]]
        # Add Nodes and their attributes
        for node_name in self.G.nodes:
            node = self.G[eval(node_name)]
            # Add Fruit colors
            if node.fruit_name is not None:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos":node.xy, "label": node.fruit_name, "color": "orange", "size": 20}
            elif node.is_target:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos": node.xy, "color": "yellow", "size": 50}
            elif node.aruco_num != -1:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos": node.xy, "label": node.aruco_num, "color": "green", "size": 20}
            elif node.is_aruco:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos": node.xy, "color": "green", "size": 20}
            elif node.is_fruit:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos": node.xy, "color": "orange", "size": 20}
            elif node.name in flattened_path and not node.is_target:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos": node.xy, "color": "red", "size": 20}

            else:
                G_img.add_node(node.name)
                node_attributes[node.name] = {"pos": node.xy, "color": "lightgrey", "size": 20}

        # Add Edges and their attributes
        for i in range(1,len(flattened_path)):
            G_img.add_edge(flattened_path[i-1], flattened_path[i])
            edge_attributes[(flattened_path[i-1], flattened_path[i])] = {"color": "red", "width": 3}
        nx.set_node_attributes(G_img, node_attributes)
        nx.set_edge_attributes(G_img, edge_attributes)
        positions = {node: data["pos"] for node, data in G_img.nodes(data=True)}
        labels = {node: data["label"] for node, data in G_img.nodes(data=True) if "label" in data}
        colors = [data["color"] for node, data in G_img.nodes(data=True)]
        sizes = [data["size"] for node, data in G_img.nodes(data=True)]
        edge_colors = [data["color"] for u, v, data in G_img.edges(data=True)]
        nx.draw_networkx_labels(G_img, pos=positions, labels=labels)
        nx.draw_networkx_nodes(G_img, pos=positions, node_color=colors, node_size=sizes)
        nx.draw_networkx_edges(G_img, pos=positions, edge_color=edge_colors)
        if self.plot:
            # Figure size
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.show(block=False)
        else:
            plt.savefig("djikstras_map.png")

    def draw_arena(self, draw_path=True) -> None:
        """draw_arena: Draw        # for fruit in fruit_pos:
        #     #  Set fruit as obstacles
        #     pass       s arena as graph"""
        G_img = nx.DiGraph()
        node_labels = {}
        # Draw Nodes
        for node_name in self.G.nodes:
            node = self.G[eval(node_name)]
            if node.aruco_num == -1:
                G_img.add_node(node.name, pos=node.xy)
            else:
                G_img.add_node(node.name, pos=node.xy)
                node_labels[node.name] = str(node.aruco_num)


        # for node_name in self.G.nodes:
        #     node = self.G[eval(node_name)]
        #     for edge in node.neighbours:
        #         G_img.add_edge(node.name, edge[0].name)
        path_edges = []
        path_colours = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        edge_colors = []
        edge_width = []

        i = 0
        for path in self.path:

            node_idx = 0
            while node_idx < len(path) - 1:
                G_img.add_edge(path[node_idx], path[node_idx + 1])
                edge_colors.append(path_colours[i])
                edge_width.append(3)
                node_idx += 1
            i += 1

        # Draw Bezier
        # t = np.linspace(0, 1, 100)
        # x_vals = []
        # y_vals = []
        # for i in t:
        #     x, y = self.bezier_curve(i)
        #     x_vals.append(x)
        #     y_vals.append(y)

        # plt.plot(x_vals, y_vals, 'r--')
        # Draw boundary
        max_x = self.arena_dimensions[0]/2
        max_y = self.arena_dimensions[1]/2
        G_img.add_node("A", pos=(-max_x, -max_y))
        G_img.add_node("B", pos=(max_x, -max_y))
        G_img.add_node("C", pos=(max_x, max_y))
        G_img.add_node("D", pos=(-max_x, max_y))

        G_img.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        # Drawing Properties
        node_colors = []
        path_nodes = []
        path_lengths = []
        for path in self.path:
            path_lengths.append(len(path))
            for node in path:
                path_nodes.append(node)
        for node in G_img.nodes:
            if node in ["A","B","C","D"]:
                node_colors.append('black')

            elif self.G[eval(node)].xy == self.G.get_nearest_node(self.location[:2]).xy:
                node_colors.append('yellow')
            elif self.G[eval(node)].is_target:
                node_colors.append('purple')
            elif self.G[eval(node)].is_fruit:
                node_colors.append('orange')
            elif self.G[eval(node)].is_obstacle:
                node_colors.append('green')

            elif node in path_nodes:
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
        node_sizes = [20 if not node in ["A","B","C","D"] else 10 for node in G_img.nodes]
        node_positions = {node: data["pos"] for node, data in G_img.nodes(data=True)}

        if draw_path:
            edge_colors = []
            edge_width = []
            path_edges = []
            for edge in G_img.edges:
                for path in self.path:
                    if edge[0] in path and edge[1] in path:
                        path_edges.append(edge)
                if edge[0] in ["A", "B", "C", "D"] or edge[1] in ["A", "B", "C", "D"]:
                    edge_colors.append("black")
                    edge_width.append(1)
                elif edge in path_edges:
                    path_colour = 0
                    index = path_edges.index(edge)
                    for i in range(len(path_lengths)):
                        if index < sum(path_lengths[:i+1]):
                            path_colour = i
                            break
                    edge_colors.append(path_colours[path_colour])
                    edge_width.append(3)

                else:
                    edge_colors.append("black")
                    edge_width.append(1)
        else:
            while len(edge_colors) < G_img.number_of_edges():
                edge_colors.append("black")
                edge_width.append(1)


        nx.draw(G_img, pos=node_positions, node_size=node_sizes, with_labels=False, node_color=node_colors, edge_color=edge_colors, width=edge_width)
        if self.plot:
            # Figure size
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.show(block=False)
        else:
            plt.savefig("djikstras_map.png")
            # Display Image
            # img = Image.open('djikstras_map.png')
        


if __name__ == '__main__':
    map_test = Map((3000, 3000), 50, true_map="est_truth_map.txt", shopping_list="shopping.txt", distance_threshold=float('inf'), aruco_size=(500,500), fruit_size=(500,500))
    map_test.generate_map()
    map_test.circle_flag = False 
    map_test.add_aruco_markers()
    map_test.add_fruits_as_obstacles()
    map_test.get_targets()
    # test_node = map_test.G.get_nearest_node((0,980))
    # test_node.fruit_name = 'test'
    # test_node.is_fruit = True
    # map_test.add_obstacles(test_node.xy,(500,500), is_aruco=True,)
    # for path in map_test.get_path_xy():
    #     print(path)
    map_test.draw_arena_v2()
    
