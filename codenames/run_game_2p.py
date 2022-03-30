import sys
import importlib
import argparse
import time
import os

from game_risk_2_players import Game2Players
from players.guesser import *
from players.codemaster import *
from utils.import_string_to_class import import_string_to_class

class GameRun:
    """Class that builds and runs a Game based on command line arguments"""

    def __init__(self):

        self.do_log = True
        self.do_print = True

        if not self.do_print:
            self._save_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        self.game_name = "default"

        self.g_kwargs = {}
        self.cm_kwargs = {}

        self.codemaster = import_string_to_class("players.codemaster_glove_rl_2p.AICodemaster")
        print('loaded codemaster class')

        self.guesser = import_string_to_class("players.guesser_w2v.AIGuesser")
        print('loaded guesser class')

        glove_vectors = Game2Players.load_glove_vecs("players/glove.6B.300d.txt")
        self.g_kwargs["glove_vecs"] = glove_vectors
        self.cm_kwargs["glove_vecs"] = glove_vectors
        print('loaded glove vectors')

        w2v_vectors = Game2Players.load_w2v("players/GoogleNews-vectors-negative300.bin")
        self.g_kwargs["word_vectors"] = w2v_vectors
        self.cm_kwargs["word_vectors"] = w2v_vectors
        print('loaded word vectors')


        # set seed so that board/keygrid can be reloaded later
        #self.seed = time.time()
        self.seed = int("3442")

    def __del__(self):
        """reset stdout if using the do_print==False option"""
        if not self.do_print:
            sys.stdout.close()
            sys.stdout = self._save_stdout


if __name__ == "__main__":
    game_setup = GameRun()

    game = Game2Players(game_setup.codemaster,
                game_setup.guesser,
                game_setup.codemaster,
                game_setup.guesser,
                seed=game_setup.seed,
                do_print=game_setup.do_print,
                do_log=game_setup.do_log,
                game_name=game_setup.game_name,
                cm1_kwargs=game_setup.cm_kwargs,
                g1_kwargs=game_setup.g_kwargs,
                cm2_kwargs=game_setup.cm_kwargs,
                g2_kwargs=game_setup.g_kwargs)

    game.run()
