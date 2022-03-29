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


class GameRisk:
    """Class that setups up game details and calls Guesser/Codemaster pair to play the game
    """

    def __init__(self, codemaster, guesser,
                 seed="time", do_print=True, do_log=True, game_name="default",
                 cm_kwargs={}, g_kwargs={}, nb_guesses=0, nb_good_guesses=0):

        self.codemaster = codemaster(**cm_kwargs)
        self.guesser = guesser(**g_kwargs)

        self.cm_kwargs = cm_kwargs
        self.g_kwargs = g_kwargs
        self.do_log = do_log
        self.game_name = game_name

        self.game_condition = GameCondition.HIT_RED
        self.game_counter = 0
        self.nb_guesses = nb_guesses
        self.nb_good_guesses = nb_good_guesses

        # self.nb_remaining_reds = 8
        # self.nb_remaining_blues = 7
        # self.nb_remaining_grays = 9


        # set seed so that board/keygrid can be reloaded later
        if seed == 'time':
            self.seed = time.time()
            random.seed(self.seed)
        else:
            self.seed = seed
            random.seed(int(seed))

        print("seed:", self.seed)

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

    def get_state(self):
        nb_remaining_reds = 8 - self.words_on_board.count("*Red*")
        nb_remaining_blues = 7 - self.words_on_board.count("*Blue*")
        nb_remaining_grays = 9 - self.words_on_board.count("*Civilian*")
        ratio = np.round(self.nb_good_guesses / self.nb_guesses, 2)
        return [nb_remaining_reds, nb_remaining_blues, nb_remaining_grays, ratio]

    def get_guesses(self):
        return self.nb_guesses, self.nb_good_guesses

    def _accept_guess(self, guess_index):
        """Function that takes in an int index called guess to compare with the key grid
        CodeMaster will always win with Red and lose if Blue =/= 7 or Assassin == 1
        """
        if self.key_grid[guess_index] == "Red":
            self.words_on_board[guess_index] = "*Red*"
            if self.words_on_board.count("*Red*") >= 8:
                return GameCondition.WIN
            return GameCondition.HIT_RED

        elif self.key_grid[guess_index] == "Blue":
            self.words_on_board[guess_index] = "*Blue*"
            if self.words_on_board.count("*Blue*") >= 7:
                return GameCondition.LOSS
            else:
                return GameCondition.CONTINUE

        elif self.key_grid[guess_index] == "Assassin":
            self.words_on_board[guess_index] = "*Assassin*"
            return GameCondition.LOSS

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
                f' R:{red_result} CM:{type(self.codemaster).__name__} '
                f'GUESSER:{type(self.guesser).__name__} SEED:{self.seed}\n'
            )

        with open("results/bot_results_new_style.txt", "a") as f:
            results = {"game_name": self.game_name,
                       "total_turns": num_of_turns,
                       "R": red_result, "B": blue_result, "C": civ_result, "A": assa_result,
                       "codemaster": type(self.codemaster).__name__,
                       "guesser": type(self.guesser).__name__,
                       "seed": self.seed,
                       "time_s": (self.game_end_time - self.game_start_time),
                       "cm_kwargs": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                     for k, v in self.cm_kwargs.items()},
                       "g_kwargs": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                    for k, v in self.g_kwargs.items()},
                       }
            f.write(json.dumps(results))
            f.write('\n')

    @staticmethod
    def clear_results():
        """Delete results folder"""
        if os.path.exists("results") and os.path.isdir("results"):
            shutil.rmtree("results")

    def step(self, action):
        words_in_play = self.get_words_on_board()
        current_key_grid = self.get_key_grid()
        self.codemaster.set_game_state(words_in_play, current_key_grid)
        done = False
        reward = 0

        # codemaster gives clue & number here
        clue, clue_num = self.codemaster.get_clue()
        self.game_counter += 1
        keep_guessing = True
        guess_num = 0
        clue_num = int(clue_num)

        self.guesser.set_clue(clue, clue_num)

        self.game_condition = GameCondition.HIT_RED
        while guess_num <= clue_num and keep_guessing and self.game_condition == GameCondition.HIT_RED:
            self.guesser.set_board(words_in_play)
            guess_answer = self.guesser.get_answer()

            # if no comparisons were made/found than retry input from codemaster
            if guess_answer is None or guess_answer == "no comparisons":
                break
            guess_answer_index = words_in_play.index(guess_answer.upper().strip())
            self.game_condition = self._accept_guess(guess_answer_index)
            self.nb_guesses += 1

            if self.game_condition == GameCondition.HIT_RED:
                guess_num += 1
                self.nb_good_guesses += 1
                keep_guessing = self.guesser.keep_guessing()

            # if guesser selected a civilian or a blue-paired word
            elif self.game_condition == GameCondition.CONTINUE:
                break

            elif self.game_condition == GameCondition.LOSS:
                self.game_end_time = time.time()
                game_counter = 25
                done = True
                reward = -25

            elif self.game_condition == GameCondition.WIN:
                self.game_end_time = time.time()
                done = True
                reward = 25 - game_counter

        next_state = self.get_state()
        return next_state, reward, done


    def run(self):
        """Function that runs the codenames game between codemaster and guesser"""
        game_condition = GameCondition.HIT_RED
        game_counter = 0
        while game_condition != GameCondition.LOSS and game_condition != GameCondition.WIN:
            # board setup and display
            print('\n' * 2)
            words_in_play = self.get_words_on_board()
            current_key_grid = self.get_key_grid()
            self.codemaster.set_game_state(words_in_play, current_key_grid)
            self._display_key_grid()
            self._display_board_codemaster()

            # codemaster gives clue & number here
            clue, clue_num = self.codemaster.get_clue()
            game_counter += 1
            keep_guessing = True
            guess_num = 0
            clue_num = int(clue_num)

            print('\n' * 2)
            self.guesser.set_clue(clue, clue_num)

            game_condition = GameCondition.HIT_RED
            while guess_num <= clue_num and keep_guessing and game_condition == GameCondition.HIT_RED:
                self.guesser.set_board(words_in_play)
                guess_answer = self.guesser.get_answer()

                # if no comparisons were made/found than retry input from codemaster
                if guess_answer is None or guess_answer == "no comparisons":
                    break
                guess_answer_index = words_in_play.index(guess_answer.upper().strip())
                game_condition = self._accept_guess(guess_answer_index)

                if game_condition == GameCondition.HIT_RED:
                    print('\n' * 2)
                    self._display_board_codemaster()
                    guess_num += 1
                    print("Keep Guessing? the clue is ", clue, clue_num)
                    keep_guessing = self.guesser.keep_guessing()

                # if guesser selected a civilian or a blue-paired word
                elif game_condition == GameCondition.CONTINUE:
                    break

                elif game_condition == GameCondition.LOSS:
                    self.game_end_time = time.time()
                    game_counter = 25
                    self._display_board_codemaster()
                    if self.do_log:
                        self.write_results(game_counter)
                    print("You Lost")
                    print("Game Counter:", game_counter)

                elif game_condition == GameCondition.WIN:
                    self.game_end_time = time.time()
                    self._display_board_codemaster()
                    if self.do_log:
                        self.write_results(game_counter)
                    print("You Won")
                    print("Game Counter:", game_counter)
