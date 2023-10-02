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

    def scale_m2pix(self, xy):
        # Max 3 -> 750
        # Min -3 -> 0
        x,y = xy
        x = 750 * (x + 3) / 6
        y = 750 * (y + 3) / 6

        return x,y

    def scale_pix2m(self, xy):
        # Max 3 -> 750
        # Min -3 -> 0
        x,y = xy
        x = 6 * (x / 750) - 3
        y = 6 * (y / 750) - 3

        return x,y

    
    def draw_state(self):
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        x,y,theta = self.state
        x,y = self.scale_m2pix([x,y])
        theta = np.rad2deg(theta)
        # for i in range(len(state)):
        #     x,y, theta = state[i]
        #     pygame.draw.circle(canvas, (255, 255, 255), (x, y), 10)
        #     if i > 0:
        #         x_1, y_1 = state[i-1][0:2]
        #         pygame.draw.line(canvas, (255, 255, 255), (x_1, y_1), (x, y), 3)
        surface = pygame.surfarray.make_surface(canvas)
        surface.convert_alpha()
        surface.fill((120,120,120,0))
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
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        bg = pygame.surfarray.make_surface(canvas)
        state_surf = self.draw_state()
        landmark_surf = self.draw_landmarks()
        waypoints_surf =  self.draw_waypoints()
        # combine surfaces
        surface = pygame.Surface((self.width, self.height))
        # Set background
        surface.blit(bg, (0, 0),special_flags=pygame.BLENDMODE_NONE)
        surface.blit(state_surf, (0, 0),special_flags=pygame.BLENDMODE_NONE)
        surface.blit(landmark_surf, (0, 0),special_flags=pygame.BLEND_RGBA_ADD)
        surface.blit(waypoints_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MAX)
        
        # Mirror surface
        surface = pygame.transform.flip(surface, False, True)
        surface = pygame.transform.rotate(surface, -90)
        
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def add_waypoint(self):
        x,y = pygame.mouse.get_pos()
        x,y = self.scale_pix2m([x,y])
        self.waypoints.append([x, y])
        # convert to m

        #self.waypoints.append(pygame.mouse.get_pos())

    def add_manual_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def draw_landmarks(self):
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        surface = pygame.surfarray.make_surface(canvas)
        surface.convert_alpha()
        surface.fill((0,0,0,0))
        for i in range(len(self.landmarks)):

            x,y = self.landmarks[i]
            x = (x) * self.m2pixel + self.width/2
            y = (y) * self.m2pixel  + self.height/2
            # Rotate images
            img = pygame.image.load(f"pics/8bit/lm_{i + 1}.png")
            img = pygame.transform.flip(img, False, True)
            surface.blit(img, (x-15, y-15))
            # Label the landmarks
            # font = pygame.font.Font('freesansbold.ttf', 12)
            # text = font.render(str(i + 1) , True, (255, 255, 255), (120, 120, 120))
            # textRect = text.get_rect()
            # textRect.center = (x, y + 20)
            # self.screen.blit(text, textRect)

        # Flip  surface
        return surface


    def draw_waypoints(self):
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        surface = pygame.surfarray.make_surface(canvas)
        surface.convert_alpha()
        surface.fill((0,0,0,0))
        for i in range(len(self.waypoints)):
            x,y = self.waypoints[i]
            x,y = self.scale_m2pix([x,y])
            pygame.draw.circle(surface, (255, 0, 0), (x,y), 10)
            # Label the waypoints
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render(str(i + 1) , True, (255, 255, 255), (120, 120, 120))
            textRect = text.get_rect()
            textRect.center = (x, y + 20)
            surface.blit(text, textRect)
            # Flip surface
        surface = pygame.transform.flip(surface, False, True)
        return surface



if __name__=="__main__":
    _, _, landmarks = read_true_map("M4_prac_map_full.txt")
    gui = GUI(750, 750)
    theta_range = np.linspace(0, 2*np.pi, 100)
    i_range = np.linspace(-3, 3, 100)
    j_range = np.linspace(-3, 3, 100)
    states = []
    for i,j,theta in zip(i_range, j_range, theta_range):
        states.append([i, j, theta])
    i = 1
    # Event Listener
    
    while True:
        gui.draw()
        time.sleep(0.1)
        gui.update_state(states[i]) 
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                gui.add_waypoint()
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        print(gui.state)
        if i < len(states)- 1 :
            i += 1
