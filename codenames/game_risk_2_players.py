import random
import time
import json
import enum
import os
import shutil
import sys

import colorama
import gensim.models.keyedvectors as word2vec
import numpy as np
from nltk.corpus import wordnet_ic

class GameCondition(enum.Enum):
    """Enumeration that represents the different states of the game"""
    HIT_RED = 0
    HIT_BLUE = 1
    HIT_ASSASSIN = 2
    LOSS = 3
    WIN = 4
    CONTINUE = 5


class Game2Players:
    """Class that setups up game details and calls Guesser/Codemaster pair to play the game
    """

    def __init__(self, codemaster1, guesser1, codemaster2, guesser2,
                 seed="time", do_print=True, do_log=True, game_name="default",
                 cm1_kwargs={}, g1_kwargs={}, cm2_kwargs={}, g2_kwargs={},
                 nb_guesses_1=0, nb_good_guesses_1=0, nb_guesses_2=0, nb_good_guesses_2=0,
                 display_board=True):
        """ Setup Game details

        Args:
            codemaster (:class:`Codemaster`):
                Codemaster (spymaster in Codenames' rules) class that provides a clue.
            guesser (:class:`Guesser`):
                Guesser (field operative in Codenames' rules) class that guesses based on clue.
            seed (int or str, optional): 
                Value used to init random, "time" for time.time(). 
                Defaults to "time".
            do_print (bool, optional): 
                Whether to keep on sys.stdout or turn off. 
                Defaults to True.
            do_log (bool, optional): 
                Whether to append to log file or not. 
                Defaults to True.
            game_name (str, optional): 
                game name used in log file. Defaults to "default".
            cm_kwargs (dict, optional): 
                kwargs passed to Codemaster.
            g_kwargs (dict, optional): 
                kwargs passed to Guesser.
        """

        self.game_start_time = time.time()
        colorama.init()

        self.do_print = do_print
        if not self.do_print:
            self._save_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        self.display_board = display_board

        self.codemasters = [codemaster1(**cm1_kwargs), codemaster2(**cm2_kwargs)]
        self.guessers = [guesser1(**g1_kwargs), guesser2(**g2_kwargs)]
        self.codemasters[0].set_player_index(0)
        self.codemasters[1].set_player_index(1)

        self.cm_kwargss = [cm1_kwargs, cm2_kwargs]
        self.g_kwargss = [g1_kwargs, g2_kwargs]
        self.do_log = do_log
        self.game_name = game_name

        self.game_condition = GameCondition.HIT_RED
        self.game_counters = [0, 0]
        self.nb_guessess = [nb_guesses_1, nb_guesses_2]
        self.nb_good_guessess = [nb_good_guesses_1, nb_good_guesses_2]

        # set seed so that board/keygrid can be reloaded later
        if seed == 'time':
            self.seed = time.time()
            random.seed(self.seed)
        else:
            self.seed = seed
            random.seed(int(seed))

        #print("seed:", self.seed)

        # load board words
        with open("game_wordpool.txt", "r") as f:
            temp = f.read().splitlines()
            assert len(temp) == len(set(temp)), "game_wordpool.txt should not have duplicates"
            random.shuffle(temp)
            self.words_on_board = temp[:25]

        # set grid key for codemaster (spymaster)
        self.key_grid = ["Red"] * 8 + ["Blue"] * 7 + ["Civilian"] * 9 + ["Assassin"]
        random.shuffle(self.key_grid)

    def __del__(self):
        """reset stdout if using the do_print==False option"""
        if not self.do_print:
            sys.stdout.close()
            sys.stdout = self._save_stdout

    @staticmethod
    def load_glove_vecs(glove_file_path):
        """Load stanford nlp glove vectors
        Original source that matches the function: https://nlp.stanford.edu/data/glove.6B.zip
        """
        with open(glove_file_path, encoding="utf-8") as infile:
            glove_vecs = {}
            for line in infile:
                line = line.rstrip().split(' ')
                glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])
            return glove_vecs

    @staticmethod
    def load_wordnet(wordnet_file):
        """Function that loads wordnet from nltk.corpus"""
        return wordnet_ic.ic(wordnet_file)

    @staticmethod
    def load_w2v(w2v_file_path):
        """Function to initalize gensim w2v object from Google News w2v Vectors
        Vectors Source: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        """
        return word2vec.KeyedVectors.load_word2vec_format(w2v_file_path, binary=True, unicode_errors='ignore')

    def _display_board_codemaster(self):
        """prints out board with color-paired words, only for codemaster, color && stylistic"""
        print(str.center("___________________________BOARD___________________________\n", 60))
        counter = 0
        for i in range(len(self.words_on_board)):
            if counter >= 1 and i % 5 == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RED + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.BLUE + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.MAGENTA + self.words_on_board[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 60))
        print("\n")

    def _display_board(self):
        """prints the list of words in a board like fashion (5x5)"""
        print(colorama.Style.RESET_ALL)
        print(str.center("___________________________BOARD___________________________", 60))
        for i in range(len(self.words_on_board)):
            if i % 5 == 0:
                print("\n")
            print(str.center(self.words_on_board[i], 10), " ", end='')

        print(str.center("\n___________________________________________________________", 60))
        print("\n")

    def _display_key_grid(self):
        """ Print the key grid to stdout  """
        print("\n")
        print(str.center(colorama.Fore.RESET +
                         "____________________________KEY____________________________\n", 55))
        counter = 0
        for i in range(len(self.key_grid)):
            if counter >= 1 and i % 5 == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RED + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.BLUE + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.MAGENTA + self.key_grid[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 55))
        print("\n")

    def get_words_on_board(self):
        """Return the list of words that represent the board state"""
        return self.words_on_board

    def get_key_grid(self):
        """Return the codemaster's key"""
        return self.key_grid

    def get_state(self, player_index):
        nb_remaining_reds = 8 - self.words_on_board.count("*Red*")
        nb_remaining_blues = 7 - self.words_on_board.count("*Blue*")
        nb_remaining_grays = 9 - self.words_on_board.count("*Civilian*")
        if self.nb_guessess[player_index] > 0:
            ratio = np.round(self.nb_good_guessess[player_index] / self.nb_guessess[player_index], 2)
        else:
            ratio = 0
        if player_index == 0:
            return [nb_remaining_reds/9, nb_remaining_blues/9, nb_remaining_grays/9, ratio]
        else:
            return [nb_remaining_blues/9, nb_remaining_reds/9, nb_remaining_grays/9, ratio]

    def get_guesses(self, player_index):
        return self.nb_guessess[player_index], self.nb_good_guessess[player_index]

    def _accept_guess(self, guess_index, player_index):
        """Function that takes in an int index called guess to compare with the key grid
        CodeMaster will always win with Red and lose if Blue =/= 7 or Assassin == 1
        """
        if self.key_grid[guess_index] == "Red":
            self.words_on_board[guess_index] = "*Red*"
            if self.words_on_board.count("*Red*") >= 8:
                return GameCondition.WIN
            return [GameCondition.HIT_RED, GameCondition.CONTINUE][player_index]

        elif self.key_grid[guess_index] == "Blue":
            self.words_on_board[guess_index] = "*Blue*"
            if self.words_on_board.count("*Blue*") >= 7:
                return GameCondition.LOSS
            else:
                return [GameCondition.CONTINUE, GameCondition.HIT_BLUE][player_index]

        elif self.key_grid[guess_index] == "Assassin":
            self.words_on_board[guess_index] = "*Assassin*"
            print("ASSASSIN !")
            return [GameCondition.LOSS, GameCondition.WIN][player_index]

        else:
            self.words_on_board[guess_index] = "*Civilian*"
            return GameCondition.CONTINUE

    def write_results(self, num_of_turns):
        """Logging function
        writes in both the original and a more detailed new style
        """
        red_result = 0
        blue_result = 0
        civ_result = 0
        assa_result = 0

        for i in range(len(self.words_on_board)):
            if self.words_on_board[i] == "*Red*":
                red_result += 1
            elif self.words_on_board[i] == "*Blue*":
                blue_result += 1
            elif self.words_on_board[i] == "*Civilian*":
                civ_result += 1
            elif self.words_on_board[i] == "*Assassin*":
                assa_result += 1
        total = red_result + blue_result + civ_result + assa_result

        if not os.path.exists("results"):
            os.mkdir("results")

        with open("results/bot_results.txt", "a") as f:
            f.write(
                f'TOTAL:{num_of_turns} B:{blue_result} C:{civ_result} A:{assa_result}'
                f' R:{red_result} CM_RED:{type(self.codemasters[0]).__name__} CM_BLUE:{type(self.codemasters[1]).__name__} '
                f'GUESSER_RED:{type(self.guessers[0]).__name__} GUESSER_BLUE:{type(self.guessers[1]).__name__} SEED:{self.seed}\n'
            )

        with open("results/bot_results_new_style.txt", "a") as f:
            results = {"game_name": self.game_name,
                       "total_turns": num_of_turns,
                       "R": red_result, "B": blue_result, "C": civ_result, "A": assa_result,
                       "codemaster_red": type(self.codemasters[0]).__name__,
                       "guesser_red": type(self.guessers[0]).__name__,
                       "codemaster_blue": type(self.codemasters[1]).__name__,
                       "guesser_blue": type(self.guessers[1]).__name__,
                       "seed": self.seed,
                       "time_s": (self.game_end_time - self.game_start_time),
                       "cm_kwargs_red": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                     for k, v in self.cm_kwargss[0].items()},
                       "g_kwargs_red": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                    for k, v in self.g_kwargss[0].items()},
                       "cm_kwargs_blue": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                     for k, v in self.cm_kwargss[1].items()},
                       "g_kwargs_blue": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                    for k, v in self.g_kwargss[1].items()},
                       }
            f.write(json.dumps(results))
            f.write('\n')

    @staticmethod
    def clear_results():
        """Delete results folder"""
        if os.path.exists("results") and os.path.isdir("results"):
            shutil.rmtree("results")

    def step(self, risk, player_index):
        # board setup and display
        if self.display_board:
            print('\n' * 2)
        words_in_play = self.get_words_on_board()
        current_key_grid = self.get_key_grid()
        self.codemasters[player_index].set_game_state(words_in_play, current_key_grid)
        if self.display_board:
            self._display_key_grid()
            self._display_board_codemaster()

        # codemaster gives clue & number here
        clue, clue_num = self.codemasters[player_index].get_clue(risk, do_print=self.display_board)
        self.game_counters[player_index] += 1
        keep_guessing = True
        guess_num = 0
        clue_num = int(clue_num)

        # RL variables initialization
        done = False
        rewards = [0, 0]

        if self.display_board:
            print('\n' * 2)
        self.guessers[player_index].set_clue(clue, clue_num, do_print=self.display_board)

        self.game_condition = [GameCondition.HIT_RED, GameCondition.HIT_BLUE][player_index]
        while guess_num <= clue_num and keep_guessing and self.game_condition == [GameCondition.HIT_RED, GameCondition.HIT_BLUE][player_index]:
            self.guessers[player_index].set_board(words_in_play)
            guess_answer = self.guessers[player_index].get_answer(do_print=self.display_board)

            # if no comparisons were made/found than retry input from codemaster
            if guess_answer is None or guess_answer == "no comparisons":
                break
            guess_answer_index = words_in_play.index(guess_answer.upper().strip())
            self.game_condition = self._accept_guess(guess_answer_index, player_index)
            self.nb_guessess[player_index] += 1

            if self.game_condition == [GameCondition.HIT_RED, GameCondition.HIT_BLUE][player_index]:
                if self.display_board:
                    print('\n' * 2)
                if self.display_board:
                    self._display_board_codemaster()
                guess_num += 1
                self.nb_good_guessess[player_index] += 1
                if self.display_board:
                    print("Keep Guessing? the clue is ", clue, clue_num)
                keep_guessing = self.guessers[player_index].keep_guessing()

            # if guesser selected a civilian or a blue-paired word
            elif self.game_condition == GameCondition.CONTINUE:
                rewards[player_index] = guess_num
                break

            elif self.game_condition == GameCondition.LOSS:
                self.game_end_time = time.time()
                self.game_counters[0] = 25
                done = True
                rewards = [-25, 25]
                if self.display_board:
                    self._display_board_codemaster()
                if self.do_log:
                    self.write_results(self.game_counters[player_index])
                print("Red Lost")
                # print("Red Game Counter:", self.game_counters[0])
                # print("Blue Game Counter:", self.game_counters[1])

            elif self.game_condition == GameCondition.WIN:
                self.game_end_time = time.time()
                self.game_counters[1] = 25
                done = True
                rewards = [25, -25]
                if self.display_board:
                    self._display_board_codemaster()
                if self.do_log:
                    self.write_results(self.game_counters[player_index])
                print("Red Won")
                # print("Red Game Counter:", self.game_counters[0])
                # print("Blue Game Counter:", self.game_counters[1])

        next_states = [self.get_state(0), self.get_state(1)]
        return next_states, rewards, done

    def run(self, risk_1=0.7, risk_2=0.7):
        """Function that runs the codenames game between codemaster and guesser"""

        player_index = 0
        risk_levels = [risk_1, risk_2]

        while self.game_condition != GameCondition.LOSS and self.game_condition != GameCondition.WIN:
            self.step(risk_levels[player_index], player_index)
            player_index = 1 - player_index

        return self.game_condition == GameCondition.WIN, self.game_counters