import pygame

# 初始化 pygame
pygame.init()

# 创建屏幕
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Keyboard Input Example')

# 游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                print("Left arrow key pressed")
            elif event.key == pygame.K_RIGHT:
                print("Right arrow key pressed")
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_RETURN:
                print("Enter key pressed")

    # 更新屏幕
    screen.fill((0, 0, 0))
    pygame.display.flip()

# 退出 pygame
pygame.quit()
