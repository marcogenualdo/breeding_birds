import pygame as pg
from random import randint
from collections import deque
from time import time
from math import sqrt
import numpy as np

#defining colors
black = (0,0,0)
red = lambda shade: (shade,0,0)
green = lambda shade: (0,shade,0)
blue = lambda shade: (0,0,shade)

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
        self.top_score = 0
    
    def isAlive (self):
        if self.life:
            return True
        return False

    def flap (self):
        self.vy += 20

    def die (self):
        self.life = False
        self.lifespan = time() - self.start_time
        if self.lifespan > self.top_score:
            self.top_score = self.lifespan

    def respawn (self):
        self.life = True
        self.y = height // 2
        self.vy = 0
        self.start_time = time()


class Brain:
    def __init__ (self, weight_Mat, biases):
        self.nLayers = len(biases)

        self.wMat = weight_Mat
        self.bias = biases

    
    def read (self,v0):
        result = [v0] #result[k] = v[k] = W[k-1]*sigma(v[k-1])+b[k-1]
        appo = v0 #appo = sigma(result[k]) = sigma(v[k]) 
        
        for k in range(self.nLayers):
            result.append(self.wMat[k].dot(appo) + self.bias[k])
            appo = sigma(result[k+1])
        return result

    
    def guess (self,x):
        for w,b in zip(self.wMat,self.bias):
            x = sigma (w.dot(x) + b)
        return x 


#activation function
sigma = lambda z: 1 / (1 + np.exp(-z))
#size of the first NN layer 
sz0 = 3 

class SmartBird (Bird, Brain):
    def __init__(self, color, w, b):
        Bird.__init__(self, color)
        Brain.__init__(self,w,b)
        
        self.dist = np.zeros(sz0)


    def norm2 (self,x,y):
        """Computes euclidean distance from self to point (x,y)"""
        return sqrt ((x - self.x)**2 + (y - self.y)**2)


    def get_dist (self, rects):
        """Computes the distances from self to the rectagles beyond self.x"""
        self.dist[0] = self.y / height

        i = 0
        while self.x > rects[i][0] + rect_width:
            i += 1

        self.dist[1] = (rects[i][1] - self.y) / height
        self.dist[2] = (rects[i][0] + rect_width - self.x) / width
         


    def act (self, rects):
        self.get_dist(rects)
        if self.guess(self.dist) > 0.5:
            self.flap()


def next_gen (birds):
    nPlayers = len(birds)

    #classifying birds based on top performace
    birds.sort (key = lambda bird: bird.top_score, reverse = True)

    #adding gaussian noise
    lead_end = 10
    for i in range(lead_end, nPlayers):
        birds[i].wMat = birds[i % lead_end].wMat \
                + np.random.randn (*np.shape(birds[i].wMat))
        birds[i].bias = birds[i % lead_end].bias \
                + np.random.randn (*np.shape(birds[i].bias))
        
        birds[i].color = blue(255 * i // nPlayers)

    #secant method
    step = 0.005
    t = 0.5
    for i in range(lead_end,lead_end + 300):
        t += step
        for j in range(birds[i].nLayers):
            birds[i].wMat[j] = t * birds[0].wMat[j] + (1 - t) * birds[i].wMat[j]
            birds[i].bias[j] = t * birds[0].bias[j] + (1 - t) * birds[i].bias[j]
        birds[i].color = red (255 * i // 350)


def check_dist (players, rects):
    """Makes a bird die if it touches a rectangle"""
    for b in players:
        if b.y < 0 or b.y > height:
            b.die()

        for p in rects:
            if ((b.x > p[0] and b.x < p[0] + rect_width)
                    and (b.y - b.size < p[1] or b.y + b.size > p[1] + ygap)):
                b.die()


def update_pos (players, rects, recty):
    for x in rects:
        x[0] -= 5 
    if rects[0][0] < -rect_width: 
        rects.popleft()
    if rects[-1][0] < xgap:
        rects.append([width, recty])

    for b in players:
        b.vy -= 1
        b.y -= b.vy


def draw (screen, players, rects):
    screen.fill(black)

    for bird in players:
        if bird.isAlive():
            pg.draw.circle(screen, bird.color, (bird.x,bird.y), bird.size) 
        
    for x in rects:
        pg.draw.rect(screen, green(155), (x[0], 0, rect_width, x[1]))
        pg.draw.rect(screen, green(155), (x[0],x[1] + ygap,rect_width,height - x[1] + ygap))
      

def allDead (players):
    for p in players:
        if p.isAlive():
            return False
    return True


def reset (players, rects, recty):
    rects.clear()
    rects.append ([width, recty])
    recty_pos = 0

    next_gen (players)
    for b in players:
        b.respawn()


def game (birds):
    window_size = (width, height)

    #initializing pygame
    pg.init()
    pg.display.set_caption('Flappy bird copy')
    screen = pg.display.set_mode(window_size)
    clock = pg.time.Clock()

    #initializing obstacles
    f = open('rects.dat','r')
    rects = deque([[width, int(f.readline().strip())]])

    #starting main loop
    running = True
    while running:
        #event handling
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            
            if event.type == pg.KEYDOWN:
                birds[0].flap()
        
        update_pos (birds, rects, int(f.readline().strip()))
        draw (screen, birds, rects)
        pg.display.update()
        clock.tick(30)

        #checking for losses
        check_dist (birds, rects)
        if allDead (birds):
            f.seek(0)
            reset (birds, rects, int(f.readline().strip()))

        #letting AIs make a decision
        for bird in birds:
            if bird.isAlive():
                bird.act(rects)

    f.close()


def play():
    """Initializes AIs and runs the game"""
    birds = []

    dims = [sz0,8,1]
    nPlayers = 1000
    for k in range(nPlayers):
        w = [np.random.randn(col,row)
                for col,row in zip(dims[1:],dims[:-1])]
        b = [np.random.randn(col) for col in dims[1:]]

        birds.append (SmartBird(blue(255 * k // nPlayers), w, b))

    game (birds)


if __name__ == '__main__':
    play()
