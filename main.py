import pickle
from itertools import combinations
from random import shuffle
import pandas as pd


def generate_anagram_dict():
    f = open('words_full.txt')
    d = {}
    lets = set('abcdefghijklmnopqrstuvwxyz\n')
    for word in f:
        word = word.lower()
        if '-' not in word and "'" not in word:
            if len(set(word) - lets) == 0 and 2 < len(word) < 9:
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

def get_starting_word(hand, anagram_dict, starting_length=10, anagram_letter=None):

    for length in range(starting_length, 2, -1):
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

        if len(viable_words):

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


class Cell:
    def __init__(self, x, y):
        self.xpos = x
        self.ypos = y

    def add_word(self, word, direction):

        if direction == 'R':
            self.word_h = word
        elif direction == 'D':
            self.word_v = word

    def add_letter(self, letter):
        self.letter = letter

def generate_board(x=100,y=100):

    df = pd.DataFrame(columns=range(-x,x), index=range(-y,y))
    for x_pos in range (-x,x):
        for y_pos in range(-y,y):
            df.at[x_pos, y_pos] = Cell(x=x_pos, y=y_pos)  # Todo coords are wrong way here (ohwell?)

    # TODO - dataframe needs to be filled with instances of Cells, with the x,y pos included within the obj
    return df

def place_word(df, word, x, y, dir):

    if dir == 'R':
        for offset, letter in enumerate(word):
            cell = df.at[y, x+offset] #Todo coords are wrong way here (ohwell?)
            cell.add_word(word=word, direction=dir)
            cell.add_letter(letter=letter)
    return df


def print_board(board):
    # TODO iterate through the board, and maybe create a new pandas array with the value, and print?

if __name__ == '__main__':
    generate_anagram_dict()
    anagram_dict = load_anagrams_dict()
    tile_bank = generate_tile_bank()
    shuffle(tile_bank)

    board = generate_board(x=10, y=10)
    hand, tile_bank = deal_starting_hand(tile_bank)

    starting_word, hand = get_starting_word(hand, anagram_dict)
    board = place_word(board, word=starting_word, x=0, y=0, dir='R')

    pd.set_option('display.width', None)

    pd.set_option('display.max_columns', None)

    print(board)
    print(starting_word, hand)

    for i in range(len(starting_word)):
        try:
            word, hand = get_starting_word(hand, anagram_dict, starting_length=5, anagram_letter = starting_word[i])
            if word is not None:
                print(word, hand, starting_word[i])
        except:
            print('error')
            pass
    print("exiting")
