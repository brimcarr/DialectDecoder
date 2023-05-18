import pygame

#%% Button class
class Button():
    def __init__(self, x, y, width, height, basecolor, text):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.basecolor = basecolor
        self.text = text
        # self.rect.topleft = (x, y)
        self.clicked = False

    def draw(self, win, outline=None):
        #Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x-2,self.y-2,self.width+4,self.height+4),0)
            
        pygame.draw.rect(win, self.basecolor, (self.x,self.y,self.width,self.height),0)
        
        if self.text != '':
            font = pygame.font.SysFont('Arial', 25)
            text = font.render(self.text, 1, (0,0,0))
            win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))
    
    def isOver(self, pos):
        #Pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
            
                return True
            
        return False

#%% DropDown class
class DropDown():

    def __init__(self, color_menu, color_option, x, y, w, h, font, main, options):
        self.color_menu = color_menu
        self.color_option = color_option
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.main = main
        self.options = options
        self.draw_menu = False
        self.menu_active = False
        self.active_option = [-1, 0]

    def draw(self, surf, scroll_y):
        pygame.draw.rect(surf, self.color_menu[self.menu_active], self.rect, 0)
        msg = self.font.render(self.main, 1, (0, 0, 0))
        surf.blit(msg, msg.get_rect(center = self.rect.center))
        if self.draw_menu:
            # Draw all rectangles here
            start_point = scroll_y
            options_short = self.options[start_point:]
            # for i, text in enumerate(self.options):
            for i, text in enumerate(options_short):
                #makes a new rectangle
                rect = self.rect.copy()
                # makes the y-coordinte of the new rectangle
                rect.y += (i+1) * self.rect.height
                # draws new rectangle with the appropriate color
                pygame.draw.rect(surf, self.color_option[1 if i == self.active_option[0] else 0], rect, 0)
                # renders appropriate class text
                msg = self.font.render(text, 1, (0, 0, 0))
                # updates the surface
                surf.blit(msg, msg.get_rect(center = rect.center))

    def update(self, event_list, scroll_y):
        mpos = pygame.mouse.get_pos()
        self.menu_active = self.rect.collidepoint(mpos)
        self.active_option = [-1,scroll_y]
        start_point = scroll_y
        options_short = self.options[start_point:]
        for i in range(len(options_short)):
            rect = self.rect.copy()
            rect.y += (i+1) * self.rect.height
            if rect.collidepoint(mpos):
                self.active_option = [i, scroll_y] 
                break

        if not self.menu_active and self.active_option[0] == -1:
            self.draw_menu = False

        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    self.draw_menu = not self.draw_menu
                elif self.draw_menu and self.active_option[0] >= 0:
                    self.draw_menu = False
                    return self.active_option
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
                scroll_y = min(scroll_y + 1, 0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 5: 
                scroll_y = max(scroll_y - 1, -(len(self.options)))
        return [-1, scroll_y] 


    
    
    
    
    