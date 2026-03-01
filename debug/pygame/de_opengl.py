import pygame
from OpenGL.GL import *

pygame.init()
screen = pygame.display.set_mode((400, 300), pygame.DOUBLEBUF | pygame.OPENGL)
glClearColor(1, 0, 0, 1)  # 红色背景
glClear(GL_COLOR_BUFFER_BIT)
pygame.display.flip()
pygame.time.wait(2000)  # 等待2秒