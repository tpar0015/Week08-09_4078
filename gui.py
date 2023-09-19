import sys, os
import cv2
import numpy as np
import time
import pygame

class GUI:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((0, 0, 0))
        pygame.display.set_caption("GUI")
        pygame.display.flip()
        self.clock = pygame.time.Clock()
        self.bg = pygame.image.load("pics/gui_mask.png")
        self.pibot_pic = pygame.image.load("pics/8bit/pibot_top.png")
        self.state = [100, 100, 0]

    def update_state(self, state):
        self.state = state

    
    def draw_state(self):
        bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((self.width,self.height, 3))*bg_rgb.astype(np.uint8)
        x,y,theta = self.state
        # for i in range(len(state)):
        #     x,y, theta = state[i]
        #     pygame.draw.circle(canvas, (255, 255, 255), (x, y), 10)
        #     if i > 0:
        #         x_1, y_1 = state[i-1][0:2]
        #         pygame.draw.line(canvas, (255, 255, 255), (x_1, y_1), (x, y), 3)
        surface = pygame.surfarray.make_surface(canvas)
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, theta), (x-50, y-50))
        return surface

    def rot_center(self, image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image
    def draw(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.bg, (0, 0))
        state_surf = self.draw_state()
        self.screen.blit(state_surf, (0, 0))

        pygame.display.flip()

    def on_click(self):
        x,y = pygame.mouse.get_pos()
        print(x,y)
        return x,y
if __name__=="__main__":
    gui = GUI(800, 600)
    theta_range = np.linspace(0, 360, 100)
    i_range = np.linspace(0, 500, 100)
    j_range = np.linspace(0, 500, 100)
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
                gui.on_click()
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if i < len(states)- 1 :
            i += 1