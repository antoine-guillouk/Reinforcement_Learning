import scipy.spatial.distance
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import itertools

from players.codemaster import Codemaster


class AICodemaster(Codemaster):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None, player_index=0):
        super().__init__()
        self.index=player_index
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.cm_wordlist = []
        with open('players/cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_wordlist.append(line.rstrip())

        self.bad_word_dists = None
        self.good_word_dists = None

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def set_player_index(self, player_index=0):
        self.index = player_index

    def get_clue(self, risk, do_print=True):
        cos_dist = scipy.spatial.distance.cosine
        good_words = []
        bad_words = []

        # Creates Red-Labeled Word arrays, and everything else arrays
        for i in range(25):
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" or self.maps[i] == ["Blue", "Red"][self.index] or self.maps[i] == "Civilian":
                bad_words.append(self.words[i].lower())
            else:
                good_words.append(self.words[i].lower())
        if do_print:
            print(f"{['RED', 'BLUE'][self.index]}:\t", good_words)

        all_vectors = (self.glove_vecs,)
        bests = {}

        if not self.bad_word_dists:
            self.bad_word_dists = {}
            for word in bad_words:
                self.bad_word_dists[word] = {}
                for val in self.cm_wordlist:
                    b_dist = cos_dist(self.concatenate(val, all_vectors), self.concatenate(word, all_vectors))
                    self.bad_word_dists[word][val] = b_dist

            self.good_word_dists = {}
            for word in good_words:
                self.good_word_dists[word] = {}
                for val in self.cm_wordlist:
                    b_dist = cos_dist(self.concatenate(val, all_vectors), self.concatenate(word, all_vectors))
                    self.good_word_dists[word][val] = b_dist

        else:
            to_remove = set(self.bad_word_dists) - set(bad_words)
            for word in to_remove:
                del self.bad_word_dists[word]
            to_remove = set(self.good_word_dists) - set(good_words)
            for word in to_remove:
                del self.good_word_dists[word]

        for clue_num in range(1, 3 + 1):
            best_per_dist = np.inf
            best_per = ''
            best_good_word = ''
            for good_word in list(itertools.combinations(good_words, clue_num)):
                best_word = ''
                best_dist = np.inf
                for word in self.cm_wordlist:
                    if not self.arr_not_in_word(word, good_words + bad_words):
                        continue

                    bad_dist = np.inf
                    worst_bad = ''
                    for bad_word in self.bad_word_dists:
                        if self.bad_word_dists[bad_word][word] < bad_dist:
                            bad_dist = self.bad_word_dists[bad_word][word]
                            worst_bad = bad_word
                    worst_good = 0
                    for good in good_word:
                        dist = self.good_word_dists[good][word]
                        if dist > worst_good:
                            worst_good = dist

                    if worst_good < best_dist and worst_good < bad_dist:
                        best_dist = worst_good
                        best_word = word
                        # print(worst_red,red_word,word)

                        if best_dist < best_per_dist:
                            best_per_dist = best_dist
                            best_per = best_word
                            best_good_word = good_word
            bests[clue_num] = (best_good_word, best_per, best_per_dist)

        if do_print:
            print("BESTS: ", bests)
        li = []
        pi = []
        chosen_clue = bests[1]
        chosen_num = 1
        for clue_num, clue in bests.items():
            best_good_word, combined_clue, combined_score = clue
            worst = -np.inf
            best = np.inf
            worst_word = ''
            for word in best_good_word:
                dist = cos_dist(self.concatenate(word, all_vectors), self.concatenate(combined_clue, all_vectors))
                if dist > worst:
                    worst_word = word
                    worst = dist
                if dist < best:
                    best = dist
            if worst < risk and worst != -np.inf:
                if do_print:
                    print(worst, chosen_clue, chosen_num)
                chosen_clue = clue
                chosen_num = clue_num

            li.append((worst / best, best_good_word, worst_word, combined_clue,
                       combined_score, combined_score ** len(best_good_word)))

        if chosen_clue[2] == np.inf:
            chosen_clue = ('', li[0][3], 0)
            chosen_num = 1
        # print("LI: ", li)
        # print("The clue is: ", li[0][3])
        if do_print:
            print('chosen_clue is:', chosen_clue)
        # return in array styled: ["clue", number]
        return chosen_clue[1], chosen_num  # [li[0][3], 1]

    def arr_not_in_word(self, word, arr):
        if word in arr:
            return False
        lemm = self.wordnet_lemmatizer.lemmatize(word)
        lancas = self.lancaster_stemmer.stem(word)
        for i in arr:
            if i == lemm or i == lancas:
                return False
            if i.find(word) != -1:
                return False
            if word.find(i) != -1:
                return False
        return True

    def combine(self, words, wordvecs):
        factor = 1.0 / float(len(words))
        new_word = self.concatenate(words[0], wordvecs) * factor
        for word in words[1:]:
            new_word += self.concatenate(word, wordvecs) * factor
        return new_word

    def concatenate(self, word, wordvecs):
        concatenated = wordvecs[0][word]
        for vec in wordvecs[1:]:
            concatenated = np.hstack((concatenated, vec[word]))
        return concatenated
