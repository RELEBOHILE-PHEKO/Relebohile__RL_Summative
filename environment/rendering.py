import pygame, numpy as np, sys

# basic colors for UI
BLACK=(10,10,10); DARK_GREY=(28,28,28); MID_GREY=(60,60,60)
LIGHT_GREY=(180,180,180); WHITE=(240,240,240)
GREEN=(0,220,80); YELLOW=(255,210,0); RED=(220,50,50)
ORANGE=(255,140,0); CYAN=(0,210,220); BLUE=(70,140,255)

SCREEN_W, SCREEN_H = 980, 620

# decide color based on vital range
def vital_colour(v, lo, hi, m=0.15):
    if lo <= v <= hi: return GREEN
    margin = (hi-lo)*m
    if (lo-margin) <= v <= (hi+margin): return YELLOW
    return RED


class ICURenderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("ICU Sepsis — RL Agent Monitor")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        # fonts for UI
        self.font_xl = pygame.font.SysFont("Consolas",46,True)
        self.font_lg = pygame.font.SysFont("Consolas",22,True)
        self.font_md = pygame.font.SysFont("Consolas",16)
        self.font_sm = pygame.font.SysFont("Consolas",13)

        # store history for waveform graphs
        self.history = {k: [] for k in ["hr","bp","o2","lactate","infection"]}
        self.max_hist = 100

        self.last_action = "—"
        self.status = "ONGOING"

    # draw a panel box
    def _panel(self, r, border=MID_GREY):
        pygame.draw.rect(self.screen, DARK_GREY, r, border_radius=8)
        pygame.draw.rect(self.screen, border, r, 2, border_radius=8)

    # draw text
    def _text(self, t, f, c, x, y, anchor="topleft"):
        s = f.render(str(t), True, c)
        self.screen.blit(s, s.get_rect(**{anchor:(x,y)}))

    # draw a vital card (HR, BP, etc.)
    def _vital(self, name, val, unit, lo, hi, plo, phi, x, y):
        col = vital_colour(val, lo, hi)
        w,h = 150,130
        r = pygame.Rect(x,y,w,h)
        self._panel(r, col)

        self._text(name, self.font_sm, LIGHT_GREY, x+8,y+6)
        self._text(f"{val:.1f}", self.font_xl, col, x+w//2,y+42,"center")
        self._text(unit, self.font_sm, LIGHT_GREY, x+w//2,y+88,"center")
        self._text(f"{lo}-{hi}", self.font_sm, MID_GREY, x+8,y+h-16)

        # small bar showing value proportionally
        bx,by,bw = x+8,y+h-7,w-16
        pygame.draw.rect(self.screen, MID_GREY, (bx,by,bw,5), border_radius=3)
        fill = int(np.clip((val-plo)/(phi-plo),0,1)*bw)
        pygame.draw.rect(self.screen, col, (bx,by,fill,5), border_radius=3)

    # draw waveform graph
    def _wave(self, hist, col, x,y,w,h,label):
        self._panel(pygame.Rect(x,y,w,h))
        self._text(label, self.font_sm, col, x+6,y+4)
        if len(hist)<2: return

        arr = np.array(hist[-w:], float)
        lo,hi = arr.min(),arr.max()
        if hi-lo<0.5: hi=lo+0.5
        norm=(arr-lo)/(hi-lo)

        pts=[(x+i*max(1,w//len(norm)), y+h-6-int(v*(h-12))) for i,v in enumerate(norm)]
        if len(pts)>1: pygame.draw.lines(self.screen, col, False, pts, 2)

    # main draw call
    def draw(self, hr,bp,o2,lac,inf,t, action="—", status="ONGOING"):
        # update history for wave graphs
        for k,v in zip(self.history.keys(), [hr,bp,o2,lac,inf]):
            self.history[k].append(v)
            if len(self.history[k])>self.max_hist:
                self.history[k].pop(0)

        self.last_action, self.status = action, status
        self.screen.fill(BLACK)

        # header
        pygame.draw.rect(self.screen, DARK_GREY,(0,0,SCREEN_W,42))
        pygame.draw.line(self.screen, MID_GREY,(0,42),(SCREEN_W,42))
        self._text("ICU SEPSIS — RL MONITOR", self.font_lg, CYAN,14,10)
        self._text(f"Step {int(t)}/50", self.font_lg, WHITE, SCREEN_W-150,10)

        # vital cards
        self._vital("HR",hr,"bpm",60,100,30,180,20,52)
        self._vital("BP",bp,"mmHg",110,130,50,200,180,52)
        self._vital("O2",o2,"%",95,100,70,100,340,52)
        self._vital("LAC",lac,"mmol",0,2,0,10,500,52)
        self._vital("INF",inf,"score",0,2,0,10,660,52)

        # progress bar
        bx,by,bw=820,62,140
        self._panel(pygame.Rect(bx,by,bw,120), BLUE)
        pct=int(t)/50
        pygame.draw.rect(self.screen, MID_GREY,(bx+10,by+30,bw-20,14))
        pygame.draw.rect(self.screen, BLUE,(bx+10,by+30,int((bw-20)*pct),14))
        self._text(f"{int(pct*100)}%", self.font_lg, BLUE, bx+bw//2,by+55,"center")

        # waveform graphs
        self._wave(self.history["hr"],vital_colour(hr,60,100),20,196,185,90,"HR")
        self._wave(self.history["bp"],vital_colour(bp,110,130),215,196,185,90,"BP")
        self._wave(self.history["o2"],vital_colour(o2,95,100),410,196,185,90,"O2")
        self._wave(self.history["lactate"],vital_colour(lac,0,2),605,196,185,90,"Lac")
        self._wave(self.history["infection"],vital_colour(inf,0,2),800,196,185,90,"Inf")

        # action box
        self._panel(pygame.Rect(20,300,580,55), BLUE)
        self._text("ACTION:", self.font_sm, LIGHT_GREY,32,308)
        self._text(action, self.font_lg, BLUE,32,323)

        # status box
        col={"ONGOING":WHITE,"RECOVERED":GREEN,"DEATH":RED,"TIMEOUT":ORANGE}.get(status,WHITE)
        self._panel(pygame.Rect(610,300,350,55), col)
        self._text("STATUS:", self.font_sm, LIGHT_GREY,622,308)
        self._text(status, self.font_lg, col,622,323)

        pygame.display.flip()
        self.clock.tick(30)

        # quit handling
        for e in pygame.event.get():
            if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                self.close(); sys.exit()

    def close(self):
        pygame.quit()