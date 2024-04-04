import random

from catanatron.game import Game
from catanatron.models.player import Player, RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

class nonAlturisticPlayer(AlphaBetaPlayer): # Assuming the AlphaBetaPlayer is "alturistic" by default
    
    def __init__(self, color, is_bot=True, altruism = 0.8):
        """Initialize the player

        Args:
            color(Color): the color of the player
            is_bot(bool): whether the player is controlled by the computer
        """
        super().__init__(color, is_bot)
        self.altruism = altruism

# Play a simple 4v4 game. Edit MyPlayer with your logic!

#game = Game(players)
#print(game.play())  # returns winning color

from catanatron.json import GameEncoder
from catanatron_gym.features import create_sample_vector, create_sample

from tqdm import tqdm
import numpy as np

from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
db = TinyDB('../Data/catanSim1.json', storage=CachingMiddleware(JSONStorage))

for i in tqdm(range(4000)):

    dictIter = {}

    r_t = np.random.beta(6, 2)
    b_t = np.random.beta(6, 2)
    o_t = np.random.beta(6, 2)
    w_t = np.random.beta(6, 2)

    dictIter['r_t'] = r_t
    dictIter['b_t'] = b_t
    dictIter['o_t'] = o_t
    dictIter['w_t'] = w_t
    
    players = [
        nonAlturisticPlayer(Color.RED, altruism=r_t),
        nonAlturisticPlayer(Color.BLUE, altruism=b_t),
        nonAlturisticPlayer(Color.ORANGE, altruism=o_t),
        nonAlturisticPlayer(Color.WHITE, altruism=w_t)
    ]

    game = Game(players)
    print(game.play())

    record = create_sample(game, Color.RED)

    dictIter = dictIter | record

    db.insert(dictIter)

db.close()