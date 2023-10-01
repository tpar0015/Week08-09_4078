import sys, os
import cv2
import numpy as np
import time
import pygame
from w8HelperFunc import *


class GUI:

    def __init__(self, width, height, map):
        self.width = width
        self.height = height
        # Create surface
        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((0, 0, 0))

        self.bg = pygame.image.load("pics/gui_mask_bg.jpg") # starry galxy background
        pygame.display.set_caption("GUI")
        # pygame.display.flip()

        self.clock = pygame.time.Clock()
        
        # Robot params
        self.pibot_pic = pygame.transform.rotate(pygame.image.load("pics/8bit/pibot_top.png"),180)
        self.state = [100, 100, 0]
        self.waypoints = []
        # _, _, self.landmarks = read_true_map(map)
        self.landmarks = [[0.0, 0.0], [.5, .5], [1, 1]]
        
        pygame.font.init()
        self.m2pixel = width / 3    # pixels / meter

    def update_state(self, state):
        self.state = state

    
    def draw_state(self):
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        x,y,theta = self.state
        x = x * self.m2pixel 
        y = y * self.m2pixel
        theta = np.rad2deg(theta)
        # for i in range(len(state)):
        #     x,y, theta = state[i]
        #     pygame.draw.circle(canvas, (255, 255, 255), (x, y), 10)
        #     if i > 0:
        #         x_1, y_1 = state[i-1][0:2]
        #         pygame.draw.line(canvas, (255, 255, 255), (x_1, y_1), (x, y), 3)
        surface = pygame.surfarray.make_surface(canvas)
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, theta), (x-15, y-15))
        return surface

    def rot_center(self, image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, -angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image
    
    def draw(self):
        # Fill the surface with sold color
        self.screen.fill((0, 0, 0))
        # draw one image onto another
        self.screen.blit(self.bg, (0, 0))
        state_surf = self.draw_state()
        # landmark_surf = self.draw_landmarks()
        self.screen.blit(state_surf, (0, 0))
        # self.screen.blit(landmark_surf, (0, 0))
        # self.draw_waypoints()

        pygame.display.flip()

    def add_waypoint(self):
        print(pygame.mouse.get_pos())
        x,y = pygame.mouse.get_pos()
        x = x / self.m2pixel # convert to m
        y = y / self.m2pixel
        self.waypoints.append([x, y])
        # convert to m

        self.waypoints.append(pygame.mouse.get_pos())

    def add_manual_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def draw_landmarks(self):
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        surface = pygame.surfarray.make_surface(canvas)
        surface = pygame.transform.flip(surface, True, False)
        for i in range(len(self.landmarks)):

            x,y = self.landmarks[i]
            x = (x) * self.m2pixel + self.width/2
            y = (y) * self.m2pixel  + self.height/2
            surface.blit(pygame.image.load(f"pics/8bit/lm_{i + 1}.png"), (x-15, y-15))
            # Label the landmarks
            # font = pygame.font.Font('freesansbold.ttf', 12)
            # text = font.render(str(i + 1) , True, (255, 255, 255), (120, 120, 120))
            # textRect = text.get_rect()
            # textRect.center = (x, y + 20)
            # self.screen.blit(text, textRect)
        return surface


    def draw_waypoints(self):
        for i in range(len(self.waypoints)):
            x,y = self.waypoints[i]
            x = x * self.m2pixel
            y = y * self.m2pixel

            pygame.draw.circle(self.screen, (255, 0, 0), (x,y), 10)
            # Label the waypoints
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render(str(i + 1) , True, (255, 255, 255), (120, 120, 120))
            textRect = text.get_rect()
            textRect.center = (x, y + 20)
            self.screen.blit(text, textRect)


if __name__=="__main__":

    gui = GUI(750, 750, "M4_prac_map_full.txt")

    # theta_range = np.linspace(0, 360, 100)
    # i_range = np.linspace(0, 500, 100)
    # j_range = np.linspace(0, 500, 100)
    # states = []
    # for i,j,theta in zip(i_range, j_range, theta_range):
    #     states.append([i, j, theta])
    # i = 1
    # Event Listener
    while True:
        gui.draw()
        # time.sleep(0.1)
        # gui.update_state(states[i]) 
        # for event in pygame.event.get():
        #     if event.type == pygame.MOUSEBUTTONDOWN:
        #         gui.add_waypoint()
        #     elif event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()
        # if i < len(states)- 1 :
        #     i += 1