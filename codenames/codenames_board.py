import numpy as np
import csv
import logging
from numpy import delete, random as random
from random import sample

logging.basicConfig(level=logging.DEBUG)
# WORDS = "words.txt"


class CodenamesGame:
    """Codenames board represents the state of the game at any given time,
    allows for construction, per team play and guesses
    """

    def __init__(
        self, custom_cards=None, custom_map=None, is_played=False, words_loc="words.txt"
    ):
        self.current_team = "red" if random.randint(2) else "blue"
        self.winning_team = None
        self.round_score = 0
        self.revealed = [False] * 25
        self.custom_cards = custom_cards
        self.custom_map = custom_map
        self.word_loc = words_loc
        if self.custom_cards is not None:
            self.codename_cards = self.custom_cards
        else:
            self.codename_cards = self.get_cards()
        if self.custom_map is not None:
            self.spymaster_map = self.custom_map
        else:
            self.spymaster_map = self.generate_map()
        # self.assassin = self.codename_cards[self.spymaster_map['assassin'][0]]
        self.remaining_cards = self.codename_cards

    def __repr__(self):
        remaining_cards = [
            c for idx, c in enumerate(self.codename_cards) if not self.revealed[idx]
        ]
        return f"remaining cards ({len(remaining_cards)}): {remaining_cards}\nspymaster map: {self.spymaster_map}"

    def get_game(self):
        return dict(
            cards=self.codename_cards,
            blue_positions=self.spymaster_map["blue"],
            red_positions=self.spymaster_map["red"],
            neutral_positions=self.spymaster_map["neutral"],
            assassin_positions=self.spymaster_map["assassin"],
            revealed=self.revealed,
            current_team=self.current_team,
        )

    def get_cards(self):
        with open(self.word_loc, "r", newline="\n") as inputfile:
            words = inputfile.read().split("\n")
        return random.choice(words, 25)

    def get_current_turn(self):
        return self.current_team

    def set_remaining_cards(self):
        self.remaining_cards = [
            c for idx, c in enumerate(self.codename_cards) if not self.revealed[idx]
        ]

    def generate_map(self):
        """Randomly creates the mapping that the spymaster will see at the start of the game

        Returns:Ò
            array -- an array of shape (25,1) representing the locations of the agents and neutrals
        """
        board_indexes = np.arange(0, 25)
        agent_numbers = dict(assassin=1, neutral=7)
        agent_numbers["red"] = 9 if self.current_team == "red" else 8
        agent_numbers["blue"] = 9 if self.current_team == "blue" else 8

        agent_positions = dict()
        for agent_type, agent_number in agent_numbers.items():
            agent_positions[agent_type], board_indexes = self.collect_agents(
                board_indexes, agent_number
            )

        return agent_positions

    def collect_agents(self, available_board_indexes, to_select):
        agents = sample(list(available_board_indexes), to_select)
        available_board_indexes = [
            bi for bi in available_board_indexes if bi not in agents
        ]
        return agents, available_board_indexes

    def make_guess(self, word):
        logging.info(f"{self.current_team} guesses {word}")
        agent_type = self.process_guess(word)
        if agent_type == "assassin":
            self.round_score -= 9
            self.winning_team = "red" if self.current_team == "blue" else "blue"
            logging.info(f"{self.winning_team} wins!")
        elif agent_type == self.current_team:
            logging.info(f"{self.current_team} guessed correctly!")
            self.round_score += 1
        elif agent_type == "neutral":
            logging.info(f"{self.current_team} tagged a neutral")
            self.next_turn()
        elif agent_type != self.current_team:
            logging.info(f"ooft, {self.current_team} revealed an enemy")
            self.round_score -= 1
            self.next_turn()
        self.set_remaining_cards()
        
        idxs = self.spymaster_map[self.current_team]
        if all([x for idx, x in enumerate(self.revealed) if idx in idxs]):
            self.winning_team = self.current_team
        return agent_type

    def process_guess(self, word):
        """Runs through the board state and marks the word as revealed, 
        and returns the card type that was revealed

        Arguments:
            word {string} -- word guessed

        Returns:
            string -- the type of agent revealed
        """
        word_idx = self.codename_cards.tolist().index(word)
        self.revealed[word_idx] = True

        for agent_type, word_list in self.spymaster_map.items():
            if word_idx in word_list:
                return agent_type

        return None

    def next_turn(self):
        self.current_team = "red" if self.current_team == "blue" else "blue"
        self.round_score = 0


#
# game = CodenamesGame(words_loc='words.txt')
# print(game)
# game
