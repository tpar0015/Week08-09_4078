"""
 -
Author: Thomas Pardy
Date Modified: 2023-08-23

"""
import numpy as np
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
    def __init__(self, arena: tuple, radius: float, true_map: str, shopping_list: str, aruco_size = (300,300), fruit_size = (300,300)):
        """
        Initializes variables
        """
        self.arena_dimensions = arena
        self.radius = radius
        self.G = Graph()
        self.location = (0,0,0)
        self.path = []
        self.object_size = (250,90)
        self.obstacle_corners = []
        self.true_map = true_map
        self.shopping_list = shopping_list
        self.aruco_size = aruco_size
        self.fruit_size = fruit_size

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
                    nodes[i][j].add_neighbour(nodes[i - 1][j], self.G.distance(nodes[i][j].xy, nodes[i-1][j].xy)) # Up
                    #nodes[i][j].add_neighbour(nodes[i - 1][j], 1) # Up
                if i < row_n - 1:
                    nodes[i][j].add_neighbour(nodes[i + 1][j], self.G.distance(nodes[i][j].xy, nodes[i+1][j].xy))  # Down
                    #nodes[i][j].add_neighbour(nodes[i + 1][j], 1)  # Down
                if j > 0:
                    nodes[i][j].add_neighbour(nodes[i][j - 1], self.G.distance(nodes[i][j].xy, nodes[i][j-1].xy)) # Left
                    #nodes[i][j].add_neighbour(nodes[i][j - 1], 1) # Left
                if j < col_n - 1:
                    nodes[i][j].add_neighbour(nodes[i][j + 1], self.G.distance(nodes[i][j].xy, nodes[i][j+1].xy)) # Right
                    #nodes[i][j].add_neighbour(nodes[i][j + 1], 1)
        # Adds nodes to graph
        for i in nodes:
            for j in i:
                self.G.add_node(j)

    def update_location(self, pose) -> None:
        """
        Updates predicted location on nodal map
        """
        self.location = pose


    def add_obstacles(self, obs_xy, object_size: tuple, is_fruit=False, is_target=False) -> None:
        """
        Re calibrates map with object blocked out on nodes.
        """
        obs_x, obs_y = obs_xy
        obs_x = obs_x + object_size[0]/2
        obs_y = obs_y + object_size[1]/2
        self.G.reset_graph()
        closest_node = self.G.get_nearest_node((obs_x, obs_y))
        obstacle_nodes = self.G.adjacent_nodes(closest_node, object_size)
        obstacle_xy = []
        for node in obstacle_nodes:
            obstacle_xy.append(node.xy)
        corners = [(min([x[0] for x in obstacle_xy]) - 1*self.radius, min([x[1] for x in obstacle_xy]) - 1*self.radius),
                    (min([x[0] for x in obstacle_xy]) - 1*self.radius, max([x[1] for x in obstacle_xy]) + 1*self.radius),
                    (max([x[0] for x in obstacle_xy]) + 1*self.radius, min([x[1] for x in obstacle_xy]) - 1*self.radius),
                    (max([x[0] for x in obstacle_xy]) + 1*self.radius, max([x[1] for x in obstacle_xy]) + 1*self.radius),
                    ]
        # corners = [(min(x[0] for x in obstacle_xy), min(x[1] for x in obstacle_xy)),
        #             (min(x[0] for x in obstacle_xy), max(x[1] for x in obstacle_xy)),
        #             (max(x[0] for x in obstacle_xy), min(x[1] for x in obstacle_xy)),
        #             (max(x[0] for x in obstacle_xy), max(x[1] for x in obstacle_xy)),
        #             ]

        self.obstacle_corners.append(corners)
        for node in obstacle_nodes:
            if is_fruit:
                node.is_fruit = True
            elif is_target:
                node.is_target = True
            self.G.set_obstacle(node)

    def add_aruco_markers(self):
        """
        Adds aruco markers to map
        """
        _, _, aruco_positions = w8.read_true_map(self.true_map)
        for aruco in aruco_positions:
            aruco = aruco * 1000
            # aruco_x = aruco[0] + self.arena_dimensions[0]/2
            # aruco_y = aruco[1] + self.arena_dimensions[1]/2
            # aruco = (aruco_x, aruco_y)
            self.add_obstacles(aruco, self.aruco_size)

    def add_fruits_as_obstacles(self):
        fruit_targets = w8.read_search_list(self.shopping_list)
        fruit_list, fruit_pos, _ = w8.read_true_map(self.true_map)
        fruit_coords = w8.print_target_fruits_pos(fruit_targets, fruit_list, fruit_pos)
        for fruit in fruit_pos:
            fruit = fruit * 1000
            self.add_obstacles(fruit, (self.fruit_size), is_fruit=True)
            # if fruit in fruit_coords:
            #     self.G.get_nearest_node(fruit).is_target = True

        # for fruit in fruit_pos:
        #     #  Set fruit as obstacles
        #     pass       


    def update_path(self, start_node, waypoint) -> None:
        """
        Updates path to avoid any new obstacles
        """
        end_node = self.G.get_nearest_node(waypoint[:2])
        print("Start Node xy: ", start_node.xy)
        if end_node is not None:
            self.G.djikstras(start_node, end_node)
            _, path = self.G.get_shortest_distance(end_node)
            if len(self.path) > 0:
                # Insert start node as first node in path
                path.insert(0,start_node.name)
            self.path.append(path)
        else:
            print("Cant find waypoint.")
        self.path[-1] = self.shorten_shortest_path(self.path[-1])

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
        return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])
    def line_intersect(self, A, B, C, D):
        """line_intersect: Returns true if line segments AB and CD intersect"""
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
    def line_obstacle_free(self, A, B):
        """
        line_obstacle_free: Returns true if line segment AB is obstacle free
        """
        
        for corner in self.obstacle_corners:
            if corner[0][1] > -750:
                print(corner)
            if self.line_intersect(A.xy, B.xy, corner[0], corner[1]) or self.line_intersect(A.xy, B.xy, corner[0], corner[2]) or self.line_intersect(A.xy, B.xy, corner[1], corner[3]) or self.line_intersect(A.xy, B.xy, corner[2], corner[3]):
                return False
        return True

    def shorten_shortest_path(self, path) -> None:
        """
        Shortens path by removing nodes that are not needed.
        """
        start_node = self.G[eval(path[0])]
        i = 1
        while i < len(path) - 1:
            current_node = self.G[eval(path[i])]
            
            # Check if line between start and current node is obstacle free
            if self.line_obstacle_free(start_node, current_node):
                path.pop(i)

            else:
                start_node = current_node
                i += 1

        return path

    def get_targets(self):
        fruit_targets = w8.read_search_list(self.shopping_list)
        fruit_list, fruit_pos, _ = w8.read_true_map(self.true_map)
        fruit_coords = w8.print_target_fruits_pos(fruit_targets, fruit_list, fruit_pos)
        for i in range(len(fruit_coords)):
            x, y = fruit_coords[i] 
            target_fruit = (x*1000, y*1000)
            print(target_fruit)
            if i == 0:
                self.update_path(self.G.get_nearest_node(self.location[:2]), target_fruit)
            else:
                self.update_path(self.G[eval(self.path[-1][-1])], target_fruit)
            self.G.reset_graph()

            
    def draw_arena(self, draw_path=True) -> None:
        """draw_arena: Draws arena as graph"""
        G_img = nx.Graph()
        # Draw Nodes
        for node_name in self.G.nodes:
            node = self.G[eval(node_name)]
            G_img.add_node(node.name, pos=node.xy)

        for node_name in self.G.nodes:
            node = self.G[eval(node_name)]
            for edge in node.neighbours:
                G_img.add_edge(node.name, edge[0].name)

        for path in self.path:
            node_idx = 0
            while node_idx < len(path) - 1:
                G_img.add_edge(path[node_idx], path[node_idx + 1])
                node_idx += 1


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
        for path in self.path:
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
                # for path in self.path:
                #     if edge[0] in path and edge[1] in path:
                #         path_edges.append(edge)
                if edge[0] in ["A", "B", "C", "D"] or edge[1] in ["A", "B", "C", "D"]:
                    edge_colors.append("black")
                    edge_width.append(1)
                elif edge in path_edges:
                    edge_colors.append("red")
                    edge_width.append(1)

                else:
                    edge_colors.append("black")
                    edge_width.append(1)
            print(path_edges)
        else:
            edge_colors = ["black" for _ in G_img.edges]
            edge_width = [1 for _ in G_img.edges]


        nx.draw(G_img, pos=node_positions, node_size=node_sizes, with_labels=False, node_color=node_colors, edge_color=edge_colors, width=edge_width)
        # Figure size
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        plt.savefig("djikstras_map.png")
        # Display Image
        img = Image.open('djikstras_map.png')
        
        img.show()


if __name__ == '__main__':
    map_test = Map((3000, 3000), 50, true_map="map/M4_prac_map_full.txt", shopping_list="M4_prac_shopping_list.txt")
    map_test.generate_map()
    map_test.add_aruco_markers()
    map_test.add_fruits_as_obstacles()
    map_test.get_targets()
    map_test.draw_arena(draw_path=True)
    
