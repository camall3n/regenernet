import numpy as np
import matplotlib.pyplot as plt
from .basicsprite import BasicSprite
from ..utils import pos2xy

class Passenger(BasicSprite):
    def __init__(self, position=(0,0), name='Alice'):
        self.position = np.asarray(position)
        self.name = name

    def plot(self, ax):
        x,y = pos2xy(self.position)+(0.5, 0.5)
        c = plt.text(x, y, self.name[0], fontsize=14,
                horizontalalignment='center', verticalalignment='center')