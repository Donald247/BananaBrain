import pickle
from itertools import combinations
from random import shuffle
import pandas as pd
import copy
import numpy as np
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)


def generate_anagram_dict():
    f = open('words.txt')
    d = {}
    lets = set('abcdefghijklmnopqrstuvwxyz\n')
    for word in f:
        word = word.lower()
        if '-' not in word and "'" not in word:
            #if len(set(word) - lets) == 0:
            word = word.strip()
            key = ''.join(sorted(word))
            if key in d:
                if word not in d[key]:
                    d[key].append(word)
            else:
                d[key] = [word]
    f.close()

    with open('anadict.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_anagrams_dict():
    with open('anadict.pickle', 'rb') as handle:
        dict = pickle.load(handle)
        return dict








def lambda_helper1(word):
    return "".join(word).lower()

def lambda_helper2(word, letter):
    return word+letter

def get_word(hand, anagram_dict, starting_length=10, anagram_letter=None, viable_len=None, word_len=None):

    if word_len is None:
        lens_to_check = range(starting_length, 1, -1)
    else:
        lens_to_check = [word_len]

    for length in lens_to_check:
        combos = list(combinations(sorted(hand), length))
        output_list = list(map(lambda_helper1, combos))
        if anagram_letter is not None:
            output_list = [anagram_letter.lower() + word for word in output_list]
            output_list = [''.join(sorted(i)) for i in output_list]
        output_list = list(set(output_list))

        viable_letters = list(set(output_list) & set(anagram_dict.keys()))


        viable_words = []
        for letters in viable_letters:
            for word in anagram_dict[letters]:
                viable_words.append(word)


        # Need to check with viable_len here

        if viable_len and len(viable_words):
            #print("prior words {}".format(len(viable_words)))
            #print(viable_words)
            letter_index_b = np.array([item.find(anagram_letter) <= viable_len[0] for item in viable_words])
            letter_index_a = np.array(
                [len(item) - item.find(anagram_letter) - 1 <= viable_len[1] for item in viable_words])
            to_keep = (letter_index_b & letter_index_a).tolist()
            viable_words = [d for (d, remove) in zip(viable_words, to_keep) if remove]
            #print("after cull words {}".format(len(viable_words)))
            #print(viable_words)

        if len(viable_words):


            #print(len(viable_words))

            word_scores = evaluate_words(viable_words)

            max_word = max(word_scores, key=word_scores.get)
            sorted_word_list = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            #if anagram_letter is not None:  # TODO - fix this, hate it!
            #    hand.append(anagram_letter.upper())
            #for letter in max_word:
            #    hand.remove(letter.upper())

            return max_word, hand, sorted_word_list  # TODO - hand doesnt need to be returned now

    #print('no words!')
    return None, hand, None

def evaluate_words(words):
    scrabble_points = bank_dict = {'A': 1,
                                   'B': 3,
                                   'C': 3,
                                   'D': 2,
                                   'E': 1,
                                   'F': 4,
                                   'G': 2,
                                   'H': 4,
                                   'I': 1,
                                   'J': 8,
                                   'K': 5,
                                   'L': 1,
                                   'M': 3,
                                   'N': 1,
                                   'O': 1,
                                   'P': 3,
                                   'Q': 10,
                                   'R': 1,
                                   'S': 1,
                                   'T': 1,
                                   'U': 1,
                                   'V': 4,
                                   'W': 4,
                                   'X': 8,
                                   'Y': 4,
                                   'Z': 10}

    word_score_dict = {}
    for word in words:
        word_score = 0
        for letter in word:
            word_score += scrabble_points[letter.upper()]

        word_score_dict[word] = word_score

    return word_score_dict


class Board:
    def __init__(self,x=10,y=10):
        board = pd.DataFrame(columns=range(-x, x), index=range(-y, y))

        char_board = pd.DataFrame(columns=range(-x, x), index=range(-y, y))
        word_v_board = pd.DataFrame(columns=range(-x, x), index=range(-y, y))
        word_h_board = pd.DataFrame(columns=range(-x, x), index=range(-y, y))
        for x_pos in range(-x, x):
            for y_pos in range(-y, y):
                #board.at[x_pos, y_pos] = Cell(x=x_pos, y=y_pos)  # Todo coords are wrong way here (ohwell?)
                char_board.at[x_pos, y_pos] = " "
                word_v_board.at[x_pos, y_pos] = None
                word_h_board.at[x_pos, y_pos] = None

        # TODO - dataframe needs to be filled with instances of Cells, with the x,y pos included within the obj


        self.letter_dict = {}
        self.added_words = []
        self.turn_counter = 0
        self.drawn_letters = []

        self.board = board
        self.char_board = char_board
        self.word_v_board = word_v_board
        self.word_h_board = word_h_board

    def place_word(self, word, x, y, dir):

        self.added_words.append((word, x, y, dir))
        if dir == 'R':
            for offset, letter in enumerate(word):
                #cell = self.board.at[y, x + offset]  # Todo coords are wrong way here (ohwell?)
                #cell.add_word(word=word, direction=dir)
                #cell.add_letter(letter=letter)

                self.word_h_board.at[y, x + offset] = word
                self.letter_dict[(x + offset, y)] = letter
                self.char_board.at[y, x + offset] = letter

        if dir == 'D':
            for offset, letter in enumerate(word):
                #cell = self.board.at[y + offset, x]  # Todo coords are wrong way here (ohwell?)
                #cell.add_word(word=word, direction=dir)
                #cell.add_letter(letter=letter)

                self.word_v_board.at[y + offset, x] = word
                self.letter_dict[(x, y + offset)] = letter
                self.char_board.at[y + offset, x] = letter

    def update_hand(self, word, anagram_letter=None):

        to_remove = list(word)
        if anagram_letter is not None:
            to_remove.remove(anagram_letter)
        for letter in to_remove:
            self.hand.remove(letter.upper())


    def remove_word(self):
        return None


    def print_board(self):

        import os
        os.system('cls')

        for index, row in self.char_board.iterrows():
            row_values = []
            for i in range(-20, 20):
                if row[i] == " ":
                    row_values.append(' ')
                else:
                    row_values.append(row[i])
            print(row_values)

        import matplotlib.pyplot as plt
        import pandas as pd
        from pandas.plotting import table

        ax = plt.subplot(111, frame_on=False)  # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        table(ax, self.char_board)  # where df is your data frame

        # render dataframe as html
        pd.set_option('display.max_colwidth', 100)
        html = self.char_board.to_html(col_space=17, justify='justify-all')

        # write html to file
        text_file = open("index.html", "w")
        text_file.write(html)
        text_file.close()

    def deal_hand(self, size=21, peel=True):
        self.hand = []
        for i in range(size):
            self.hand.append(self.bank.pop())
        if peel:
            self.turn_counter += 1
            self.drawn_letters.append(self.hand[0])
            print(self.turn_counter)
            from pygame import mixer

            mixer.init()  # you must initialize the mixer
            alert = mixer.Sound('ding.wav')
            alert.play()

    def generate_tile_bank(self):
        self.bank = []
        bank_dict = {'A': 13,
                     'B': 3,
                     'C': 3,
                     'D': 6,
                     'E': 18,
                     'F': 3,
                     'G': 4,
                     'H': 3,
                     'I': 12,
                     'J': 2,
                     'K': 2,
                     'L': 5,
                     'M': 3,
                     'N': 8,
                     'O': 11,
                     'P': 3,
                     'Q': 2,
                     'R': 9,
                     'S': 6,
                     'T': 9,
                     'U': 6,
                     'V': 3,
                     'W': 3,
                     'X': 2,
                     'Y': 3,
                     'Z': 2}

        for letter, count in bank_dict.items():
            for i in range(count):
                self.bank.append(letter)
        shuffle(self.bank)


def compute_viable_length(string, char=" "):
    return len(string) - len(string.lstrip(char))

def determine_viable_len(Board,cord):
    viable_len = (10, 10)

    word_v = Board.word_v_board[cord[0]][cord[1]]
    word_h = Board.word_h_board[cord[0]][cord[1]]

    if word_h is not None and word_v is None:
        #print("placing vertical")

        main_line_b = "".join(Board.char_board.loc[:cord[1] - 1, cord[0]].tolist()[::-1])
        main_line_a = "".join(Board.char_board.loc[cord[1] + 1:, cord[0]].tolist())

        side_1_b = "".join(Board.char_board.loc[:cord[1] - 1, cord[0] - 1].tolist()[::-1])
        side_1_a = "".join(Board.char_board.loc[cord[1] + 1:, cord[0] - 1].tolist())

        side_2_b = "".join(Board.char_board.loc[:cord[1] - 1, cord[0] + 1].tolist()[::-1])
        side_2_a = "".join(Board.char_board.loc[cord[1] + 1:, cord[0] + 1].tolist())

    elif word_v is not None and word_h is None:
        #print("placing hoz")

        main_line_b = "".join(Board.char_board.loc[cord[1], :cord[0] - 1].tolist()[::-1])
        main_line_a = "".join(Board.char_board.loc[cord[1], cord[0] + 1:].tolist())

        side_1_b = "".join(Board.char_board.loc[cord[1] + 1, :cord[0] - 1].tolist()[::-1])
        side_1_a = "".join(Board.char_board.loc[cord[1] + 1, cord[0] + 1:].tolist())

        side_2_b = "".join(Board.char_board.loc[cord[1] - 1, :cord[0] - 1].tolist()[::-1])
        side_2_a = "".join(Board.char_board.loc[cord[1] - 1, cord[0] + 1:].tolist())

    else:
        return viable_len

    main = (max(compute_viable_length(main_line_b) - 1, 0), max(compute_viable_length(main_line_a) - 1, 0))
    side_1 = (compute_viable_length(side_1_b), compute_viable_length(side_1_a))  # these sides need to have an offset!
    side_2 = (compute_viable_length(side_2_b), compute_viable_length(side_2_a))

    viable_len = (min(main[0], side_1[0], side_2[0]), (min(main[1], side_1[1], side_2[1])))
    return viable_len

def attempt_to_place(Board):

    placed_word = False
    board_save = copy.deepcopy(Board)

    word_len_list_iterator = range(5,0,-1)
    letter_list_iterator = list(Board.letter_dict.items()).copy()
    word_len_counter = 0
    letter_counter = 0

    while word_len_counter < len(word_len_list_iterator):
        word_len = word_len_list_iterator[word_len_counter]
        #print(word_len)
        word_len_counter += 1
        letter_counter = 0
        while letter_counter < len(letter_list_iterator):
            cord, letter = letter_list_iterator[letter_counter]
            letter_counter += 1
            #print(letter)
    #for word_len in range(5,0,-1):
    #    for cord, letter in list(Board.letter_dict.items()).copy():

            #cell = Board.board[cord[0]][cord[1]]
            viable_len = determine_viable_len(Board, cord)
            #Board.word_h_board[cord[0]][cord[1]]

            word_v = Board.word_v_board[cord[0]][cord[1]]
            word_h = Board.word_h_board[cord[0]][cord[1]]

            # Check to see if you should try place a word here
            if word_h is not None and word_v is not None:
                pass  # letter already has two word

            else:
                anagram_letter = letter
                best_word, Board.hand, word_list = get_word(Board.hand, anagram_dict, starting_length=7, anagram_letter=anagram_letter, viable_len=viable_len, word_len=word_len)

                # Instead iterating over the word list (vs best word)
                if word_list is not None:

                    for word,value in word_list:
                        offset = word.find(anagram_letter)  # this only does first occourance
                        viable_board = True

                        placed_word = False

                        if word_h is None:
                            # Place it horizontally
                            x0 = cord[0] - offset
                            y0 = cord[1]
                            Board.place_word(word=word, x=x0, y=y0, dir='R')
                            Board.update_hand(word=word, anagram_letter=anagram_letter)
                            placed_word = True

                        elif word_v is None:
                            # place vertically
                            x0 = cord[0]
                            y0 = cord[1] - offset
                            Board.place_word(word=word, x=x0, y=y0, dir='D')
                            Board.update_hand(word=word, anagram_letter=anagram_letter)
                            placed_word = True

                        if placed_word and len(Board.hand) > 0:
                            #print("I placed the word {}".format(word))
                            #print("Remaining hand is {}".format(Board.hand))
                            #print("Now I would try place another word")
                            #Board.print_board()

                            viable_board = attempt_to_place(Board)

                        elif placed_word and len(Board.hand) == 0:
                            #print("I placed the word {}".format(word))
                            #print("Remaining hand is empty, see {}!".format(Board.hand))
                            #print("Now I would pick up another letter")
                            Board.print_board()

                            Board.deal_hand(1)
                            #print("Remaining hand is {}".format(Board.hand))
                            print(Board.turn_counter)

                            viable_board = attempt_to_place(Board)
                        if not viable_board:
                            #print("Removing the word: {}".format(word))


                            if Board.turn_counter != board_save.turn_counter:
                                print(Board.turn_counter, board_save.turn_counter)
                                print('update hand with drawn letters')
                                board_save.hand += Board.drawn_letters[board_save.turn_counter:Board.turn_counter]
                                print('added {} to hand (drew these letters!)'.format(Board.drawn_letters[board_save.turn_counter:Board.turn_counter]))
                                print("hand is: {}".format(board_save.hand))
                                board_save.turn_counter = Board.turn_counter
                                board_save.bank = Board.bank
                                board_save.drawn_letters = Board.drawn_letters
                                # TODO - add in the drawn letters here
                                word_len_counter = 0
                                letter_counter = 0

                            Board = copy.deepcopy(board_save)
                            #Board.print_board()


                            # TODO - also need to start searching from the start again (start this loop over)

                            placed_word = False

    if not placed_word:
        #print("I couldn't place any words")
        #print("Now I would undo my previous placement and try again!")
        #Board.deal_hand(1)
        return False
    else:
        print('completed the game, exiting?')
        return True

if __name__ == '__main__':
    generate_anagram_dict()
    anagram_dict = load_anagrams_dict()

    Board = Board(x=20, y=20)
    Board.generate_tile_bank()
    Board.deal_hand(size=21, peel=False)


    starting_word, Board.hand, word_list = get_word(Board.hand, anagram_dict, starting_length=6)

    Board.place_word(word=starting_word, x=0, y=0, dir='R')
    Board.update_hand(word=starting_word)

    Board.print_board()

    attempt_to_place(Board)

    #Board.remove_word(Board.added_words[-1][0])
    print('final hand')
    print(Board.hand)
