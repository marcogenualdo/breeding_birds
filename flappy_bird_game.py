import pygame as pg
from random import randint
from collections import deque
from time import time

import numpy as np


#defining colors
black = (0,0,0)
blue = (0,0,255)
green = (0,155,0)

#sizes
width, height = 600, 400
rect_width = 100
def_x = 150
xgap = 300
ygap = 150 


class Bird:
    def __init__(self, color):
        self.size = 15
        self.color = tuple(color)
        self.x = def_x 
        self.respawn() 
    
    def isAlive (self):
        if self.life:
            return True
        return False

    def flap (self):
        self.vy += 20

    def die (self):
        self.life = False
        self.lifespan = time() - self.start_time

    def respawn (self):
        self.life = True
        self.y = height // 2
        self.vy = 0
        self.start_time = time()


class Game:
    def __init__ (self):
        self.quit = False
        self.bird = Bird(blue)
        
        #setting window size      
        self.window_size = (width, height)

        #initializing pygame
        pg.init()
        pg.display.set_caption('Flappy bird copy')
        self.screen = pg.display.set_mode(self.window_size)
        self.clock = pg.time.Clock()

        #setting variables
        self.background = pg.image.load('assets/images/background.jpeg')
        self.background = pg.transform.scale(self.background, self.window_size)

        self.rect_up_img = pg.image.load('assets/images/tp.png')
        self.rect_up_img = pg.transform.scale(self.rect_up_img, (rect_width, self.rect_up_img.get_height()))

        self.rect_down_img = pg.image.load('assets/images/bp.png')
        self.rect_down_img = pg.transform.scale(self.rect_down_img, (rect_width, self.rect_down_img.get_height()))

        self.rects = deque([[width, randint(ygap, height - ygap)]])


    def check_dist (self):
        # bird has hit the floor / ceiling
        if self.bird.y < 0 or self.bird.y > height:
            self.bird.die()

        # bird has hit one rectangle
        for p in self.rects:
            if ((self.bird.x > p[0] and self.bird.x < p[0] + rect_width)
                    and (self.bird.y - self.bird.size < p[1] or self.bird.y + self.bird.size > p[1] + ygap)):
                self.bird.die()


    def update_pos (self):
        # shifting rectagnles left
        for x in self.rects:
            x[0] -= 5 
        if self.rects[0][0] < -rect_width: 
            self.rects.popleft()
        if self.rects[-1][0] < xgap:
            self.rects.append([width, randint(0, height - ygap)])

        # making the bird fall
        self.bird.vy -= 1
        self.bird.y -= self.bird.vy


    def draw (self):
        #self.screen.fill(black)
        self.screen.blit(self.background, (0,0))

        if self.bird.isAlive():
            pg.draw.circle(
                self.screen, 
                self.bird.color, 
                (self.bird.x,self.bird.y), 
                self.bird.size
            ) 
            
        for x in self.rects:
            #pg.draw.rect(self.screen, green, (x[0], 0, rect_width, x[1]))
            #pg.draw.rect(self.screen, green, (x[0],x[1] + ygap,rect_width,height - x[1] + ygap))

            rect_up_draw = pg.transform.scale(self.rect_up_img, (rect_width, x[1]))
            self.screen.blit(rect_up_draw, (x[0], 0))

            rect_down_draw = pg.transform.scale(self.rect_down_img, (rect_width, height - x[1] + ygap))
            self.screen.blit(rect_down_draw, (x[0],x[1] + ygap))
          

    def reset (self):
        self.rects.clear()
        self.rects.append ([width, randint(0, height - ygap)])
        self.bird.respawn()
        
        return np.array([
                    self.bird.y,
                    self.bird.vy,
                    self.rects[0][0],
                    self.rects[0][1]
                ]) 

       
    def step(self, action, frames_to_skip):
        # event handling
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit = True
           
            if event.type == pg.KEYDOWN:
                self.bird.flap()
        if action:
            self.bird.flap()

        # game loop
        done = False
        reward = 0
        while self.bird.isAlive() and reward <= frames_to_skip:
            self.update_pos()
            self.draw()
            pg.display.update()
            self.clock.tick(30)

            # checking for losses
            self.check_dist()
            if not self.bird.isAlive():
                done = True
            reward += 1

        return np.array([
                self.bird.y,
                self.bird.vy,
                self.rects[0][0],
                self.rects[0][1]
            ]), \
            reward, \
            done, \
            {quit} \


if __name__ == "__main__":
    gg = Game()
    while not gg.quit:
        done = False
        state = gg.reset()
        while not done and not gg.quit:
            state, reward, done, info = gg.step(0, 0)
