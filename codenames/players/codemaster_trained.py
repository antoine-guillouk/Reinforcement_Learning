import scipy.spatial.distance
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import itertools
from embedding_rl import Trainer

from players.codemaster import Codemaster

class AICodemaster(Codemaster):

    def __init__(self, master_vecs=None, guesser_vecs=None, brown_ic=None, glove_vecs=None, word_vectors=None):
        super().__init__()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.cm_wordlist = []
        with open('players/reduced_cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_wordlist.append(line.rstrip())

        self.bad_word_dists = None
        self.red_word_dists = None

        self.master_vecs = master_vecs

        print("Training...")
        self.trainer = Trainer(master_vecs, guesser_vecs)
        self.trainer.train()

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_clue(self):
        red_words = []
        bad_words = []

        # Creates Red-Labeled Word arrays, and everything else arrays
        for i in range(25):
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
                bad_words.append(self.words[i].lower())
            else:
                red_words.append(self.words[i].lower())
        print("RED:\t", red_words)

        chosen_clue_index, chosen_proba = self.trainer.choose_clue(self.master_vecs[red_words[0]], 0)
        choose = self.trainer.check_clue(bad_words, chosen_clue_index, chosen_proba)
        avoid = [chosen_clue_index]
        while not choose:
            print("wrong")
            chosen_clue_index, chosen_proba = self.trainer.choose_clue(self.master_vecs[red_words[0]], 0, avoid)
            choose = self.trainer.check_clue(bad_words, chosen_clue_index, chosen_proba)
            avoid.append(chosen_clue_index)
        
        chosen_clue = self.trainer.clue_to_word(chosen_clue_index)

        # print("LI: ", li)
        # print("The clue is: ", li[0][3])
        print('chosen_clue is:', chosen_clue)
        # return in array styled: ["clue", number]
        return chosen_clue, 1  # [li[0][3], 1]

