import pickle
from itertools import combinations
from random import shuffle
import pandas as pd


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


def generate_tile_bank():
    bank = []
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
            bank.append(letter)

    return bank


def deal_starting_hand(bank, size=21):
    hand = []
    for i in range(size):
        hand.append(bank.pop())

    return hand, bank


def lambda_helper1(word):
    return "".join(word).lower()

def lambda_helper2(word, letter):
    return word+letter

def get_word(hand, anagram_dict, starting_length=10, anagram_letter=None, viable_len=None):

    for length in range(starting_length, 0, -1):
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
            print("prior words {}".format(len(viable_words)))
            print(viable_words)
            import numpy as np
            letter_index_b = np.array([item.find(anagram_letter) <= viable_len[0] for item in viable_words])
            letter_index_a = np.array(
                [len(item) - item.find(anagram_letter) - 1 <= viable_len[1] for item in viable_words])
            to_keep = (letter_index_b & letter_index_a).tolist()
            viable_words = [d for (d, remove) in zip(viable_words, to_keep) if remove]
            print("after cull words {}".format(len(viable_words)))
            print(viable_words)

        if len(viable_words):


            print(len(viable_words))

            word_scores = evaluate_words(viable_words)

            max_word = max(word_scores, key=word_scores.get)

            if anagram_letter is not None:  # TODO - fix this, hate it!
                hand.append(anagram_letter.upper())
            for letter in max_word:
                hand.remove(letter.upper())

            return max_word, hand

    print('no words!')
    return None, hand

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

        for x_pos in range(-x, x):
            for y_pos in range(-y, y):
                board.at[x_pos, y_pos] = Cell(x=x_pos, y=y_pos)  # Todo coords are wrong way here (ohwell?)
                char_board.at[x_pos, y_pos] = " "
        # TODO - dataframe needs to be filled with instances of Cells, with the x,y pos included within the obj

        self.board = board
        self.char_board = char_board
        self.letter_dict = {}

    def place_word(self, word, x, y, dir):
        if dir == 'R':
            for offset, letter in enumerate(word):
                cell = self.board.at[y, x + offset]  # Todo coords are wrong way here (ohwell?)
                cell.add_word(word=word, direction=dir)
                cell.add_letter(letter=letter)

                self.letter_dict[(x + offset, y)] = letter
                self.char_board.at[y, x + offset] = letter
        if dir == 'D':
            for offset, letter in enumerate(word):
                cell = self.board.at[y + offset, x]  # Todo coords are wrong way here (ohwell?)
                cell.add_word(word=word, direction=dir)
                cell.add_letter(letter=letter)

                self.letter_dict[(x, y + offset)] = letter
                self.char_board.at[y + offset, x] = letter


    def print_board(self):
        for index, row in self.board.iterrows():
            row_values = []
            for i in range(-20, 20):
                if row[i].letter is None:
                    row_values.append(' ')
                else:
                    row_values.append(row[i].letter)
            print(row_values)

class Cell:
    def __init__(self, x, y):
        self.xpos = x
        self.ypos = y
        self.letter = None
        self.word_h = None
        self.word_v = None

    def add_word(self, word, direction):

        if direction == 'R':
            self.word_h = word
        elif direction == 'D':
            self.word_v = word

    def add_letter(self, letter):
        self.letter = letter


def compute_viable_length(string, char=" "):
    return len(string) - len(string.lstrip(char))

if __name__ == '__main__':
    generate_anagram_dict()
    anagram_dict = load_anagrams_dict()
    tile_bank = generate_tile_bank()
    shuffle(tile_bank)

    Board = Board(x=20, y=20)

    hand, tile_bank = deal_starting_hand(tile_bank)

    starting_word, hand = get_word(hand, anagram_dict, starting_length=5)

    Board.place_word(word=starting_word, x=0, y=0, dir='R')
    Board.print_board()
    #pd.set_option('display.width', None)

    #pd.set_option('display.max_columns', None)

    #print(board)
    #print(starting_word, hand)

    #for i in range(len(starting_word)):

    complete = False
    while not complete:
        print('looping')
        placed_word = False

        for cord, letter in Board.letter_dict.items():

            cell = Board.board[cord[0]][cord[1]]
            print('')

            viable_len = (10,10)
            if cell.word_h is not None and cell.word_v is None:
                print("placing vertical")
                print(letter)


                main_line_b = "".join(Board.char_board.loc[:cord[1]-1, cord[0]].tolist()[::-1])
                main_line_a = "".join(Board.char_board.loc[cord[1]+1:, cord[0]].tolist())

                side_1_b = "".join(Board.char_board.loc[:cord[1]-1,cord[0]-1].tolist()[::-1])
                side_1_a = "".join(Board.char_board.loc[cord[1]+1:,cord[0]-1].tolist())

                side_2_b = "".join(Board.char_board.loc[:cord[1]-1,cord[0]+1].tolist()[::-1])
                side_2_a = "".join(Board.char_board.loc[cord[1]+1:,cord[0]+1].tolist())

            if cell.word_v is not None and cell.word_h is None:
                print("placing hoz")
                print(letter)
                main_line_b = "".join(Board.char_board.loc[cord[1], :cord[0]-1].tolist()[::-1])
                main_line_a = "".join(Board.char_board.loc[cord[1], cord[0]+1:].tolist())

                side_1_b = "".join(Board.char_board.loc[cord[1]+1, :cord[0]-1].tolist()[::-1])
                side_1_a = "".join(Board.char_board.loc[cord[1]+1, cord[0]+1:].tolist())

                side_2_b = "".join(Board.char_board.loc[cord[1]-1, :cord[0]-1].tolist()[::-1])
                side_2_a = "".join(Board.char_board.loc[cord[1]-1, cord[0]+1:].tolist())


            main = (max(compute_viable_length(main_line_b)-1,0), max(compute_viable_length(main_line_a)-1, 0))
            side_1 = (compute_viable_length(side_1_b), compute_viable_length(side_1_a)) # these sides need to have an offset!
            side_2 = (compute_viable_length(side_2_b), compute_viable_length(side_2_a))

            viable_len = (min(main[0], side_1[0], side_2[0]), (min(main[1], side_1[1], side_2[1])))
            print(viable_len)
            #print('wait')

            # Determine the viable word sizes here:

            # Determine if placing up or right
            # Get list of all characters along that line, and either side (create a pandas array with chars to make this quick)


            # This can probs be done with some smart list searches for non "" across the entire row/col
            # for the main line (with letter a as focus):
                # ["","o","","","a","","","","","o"] -> x a x x x

            # for the side lines (with letter a as focus on the center line):
                # ["","g","","","a","","","g","",""] -> x x a x x
                # different since they can touch on diagonals

            # Then somehow combine these:
                # x a x x x -> 01000 (1b, 3)
                # x x a x x -> 00100  (2b, 2a)
                # x x x x x a -> 000001  (5b, 0a)
                # min(1,2,3)b, min(3,2,0)a = 1b, 0a?

            # Now somehow filter all words that fit with this 1b 0a template
            # Only pick from these subset of words that wont cause cross overs!

            # Check to see if you should try place a word here
            if cell.word_h is not None and cell.word_v is not None:
                print('dont test this letter!')

                pass  # letter already has two word
            #elif [Board.board[cord[0]+1][cord[1]+1].letter, Board.board[cord[0]+1][cord[1]-1].letter,
            #      Board.board[cord[0]-1][cord[1]+1].letter,Board.board[cord[0]-1][cord[1]-1].letter] != [None] * 4:
            #    print('dont test this letter! (nearby word')
            #    pass  #

            else:

                anagram_letter = letter
                word, hand = get_word(hand, anagram_dict, starting_length=5, anagram_letter = anagram_letter, viable_len = viable_len)

                if word is not None:
                    print(word, hand, anagram_letter)
                    #Board.place_word(word=starting_word, x=0, y=0, dir='R')
                    offset = word.find(anagram_letter)  #this only does first occourance

                    if cell.word_h is None:
                        # Place it horizontally
                        x0 = cord[0] - offset
                        y0 = cord[1]
                        Board.place_word(word=word, x=x0, y=y0, dir='R')
                        Board.print_board()

                    elif cell.word_v is None:
                        # place vertically
                        x0 = cord[0]
                        y0 = cord[1] - offset
                        Board.place_word(word=word, x=x0, y=y0, dir='D')
                        Board.print_board()

                    print('breaking')
                    placed_word = True
                    break

        if not placed_word or len(hand)==0:
            complete=True
    print("exiting")

    print('final hand')
    print(hand)
