from os import getcwd, chdir

import pandas as pd
import numpy as np
import requests
import zipfile
import plotly.express as px
import plotly.graph_objs as go
import io
import gensim.downloader as api
import logging

from collections import namedtuple
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import Normalizer
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
from itertools import combinations
from codenames.codenames_board import CodenamesGame
from tqdm.auto import tqdm
from os.path import isfile, join
from sklearn.metrics.pairwise import cosine_similarity as cosine
from operator import itemgetter

logger = logging.getLogger()


def restrict_vocab_to_english(model):
    (
        english_only_vocab,
        english_only_vectors,
        english_only_index2entity,
        english_only_vectors_norm,
    ) = ({}, [], [], [])

    for word, vector, index2entity in zip(
        model.vocab.keys(), model.vectors, model.index2entity
    ):

        if "/c/en/" in word:
            vocab = model.vocab[word]

            vocab.index = len(english_only_index2entity)
            english_only_vocab[word.replace("/c/en/", "")] = vocab
            english_only_vectors.append(vector)
            english_only_index2entity.append(index2entity.replace("/c/en/", ""))

    model.vocab = english_only_vocab
    model.vectors = np.array(english_only_vectors)
    model.index2entity = english_only_index2entity
    model.index2word = english_only_index2entity


def get_game_data(codenames_game, model):
    team = codenames_game.current_team
    enemy_team = "blue" if team == "red" else "blue"
    spymaster_map = codenames_game.spymaster_map
    revealed = codenames_game.revealed
    cards = [smart_lower(card, model) for card in codenames_game.codename_cards]
    in_play_cards = cards.copy()
    ally_cards = [
        c for i, c in enumerate(cards) if i in spymaster_map[team] and not revealed[i]
    ]
    ally_cards = [smart_lower(ally_card, model) for ally_card in ally_cards]
    enemy_cards = [
        c
        for i, c in enumerate(cards)
        if i in spymaster_map[enemy_team] and not revealed[i]
    ]
    enemy_cards = [smart_lower(enemy_card, model) for enemy_card in enemy_cards]
    neutral_cards = [
        c
        for i, c in enumerate(cards)
        if i in spymaster_map["neutral"] and not revealed[i]
    ]
    neutral_cards = [smart_lower(neutral_card, model) for neutral_card in neutral_cards]
    assassin_card = cards[spymaster_map["assassin"][0]]
    cards = [c for i, c in enumerate(cards) if not revealed[i]]
    return (
        cards,
        ally_cards,
        enemy_cards,
        neutral_cards,
        assassin_card,
        spymaster_map,
        team,
    )


def smart_lower(word, model):
    checked = False
    adjusted = word.replace(" ", "").lower()
    try:
        model.get_vector(adjusted)
    except KeyError:
        adjusted = adjusted.capitalize()
        model.get_vector(adjusted)

    return adjusted


def create_word_combinations_matrices(
    ally_cards, model, default_max_combo=5, one_word_clues=False
):
    max_combination = min(len(ally_cards), default_max_combo)
    ally_combinations = []
    lowest_combination = 1 if (one_word_clues or len(ally_cards) == 1) else 2
    for i in range(lowest_combination, max_combination + 1):
        ally_combinations += list(combinations(ally_cards, i))

    ally_combination_vectors = [
        [model.get_vector(word) for word in combination]
        for combination in ally_combinations
    ]
    return ally_combinations, ally_combination_vectors


def get_most_similar(
    positive_cards, all_cards, model, negative_cards=None, include_score=False, topn=1
):
    try:
        most_similar_words_with_scores = model.most_similar(
            positive=positive_cards, negative=negative_cards, topn=50,
        )
        possible_clue_words = []
        for most_similar_word_w_score in most_similar_words_with_scores:
            word = most_similar_word_w_score[0]
            if all([[c not in word for c in all_cards]]):
                possible_clue_words.append(word)

        return possible_clue_words[:topn]
    except IndexError:
        logger.critical(f"No valid clue found!\nCombination was {positive_cards}")
        logger.critical


def get_most_similar_vectors_for_combos(word_combinations, all_cards, model, **kwargs):
    most_similar_words = {
        combination: get_most_similar(
            positive_cards=list(combination), all_cards=all_cards, model=model, **kwargs
        )
        for combination in word_combinations
    }

    return most_similar_words


ClueTuple = namedtuple("ClueTuple", ["clue", "intended_combo", "board_similarities"])


def create_clue_tuples(word_combo_clue_dict, cards, model):
    clue_tuples = []
    for clue_words, clues in word_combo_clue_dict.items():
        for clue in clues:
            clue_tuples.append(
                ClueTuple(
                    clue,
                    clue_words,
                    [(card, model.similarity(card, clue)) for card in cards],
                )
            )

    return clue_tuples


def create_clue_df(clue_tuples, cards):
    dataframe_tuples = []
    for clue, intended_combo, board_similarities in clue_tuples:
        dataframe_tuples.append(
            (
                clue,
                intended_combo,
                *[card_similarity[1] for card_similarity in board_similarities],
            )
        )

    return pd.DataFrame(
        dataframe_tuples, columns=["clue", "intended_combo"] + cards
    ).set_index(["clue", "intended_combo"])


def calculate_best_clue(
    clue_df,
    spymaster_map,
    ally_cards,
    enemy_cards,
    neutral_cards,
    assassin_card,
    **kwargs,
):
    assassin_weight = kwargs.get("assassin_weight", -10)
    enemy_weight = kwargs.get("enemy_weight", -5)
    neutral_weight = kwargs.get("neutral_weight", 0)
    ally_weight = kwargs.get("ally_weight", 10)
    risk_weight = kwargs.get("risk_weight", 0)
    clue_score_threshold = kwargs.get("clue_score_threshold", 0)
    with_normalisation = kwargs.get("with_normalisation", False)

    ally_cards_len = len(ally_cards)
    weighted_clue_df = clue_df.copy()
    if with_normalisation:
        norm = Normalizer()
        weighted_clue_df.iloc[:, :25] = norm.fit_transform(
            weighted_clue_df.iloc[:, :25]
        )
    weighted_clue_df["raw_clue_length"] = weighted_clue_df.index.get_level_values(
        "intended_combo"
    ).str.len()
    # clip scores below the threshold
    weighted_clue_df.loc[:, ally_cards] = (
        ally_df := weighted_clue_df.loc[:, ally_cards]
    ).where(ally_df >= clue_score_threshold, 0)
    # If clue word score is part of intended, but below threshold we reduce the clue length!
    weighted_clue_df["amended_combo"] = create_amended_combos(weighted_clue_df)
    weighted_clue_df["actual_combo_length"] = weighted_clue_df.amended_combo.str.len()

    # Apply various weights
    weighted_clue_df.loc[:, [assassin_card]] = (
        weighted_clue_df.loc[:, [assassin_card]] * assassin_weight
    )
    weighted_clue_df.loc[:, neutral_cards] = (
        weighted_clue_df.loc[:, neutral_cards] * neutral_weight
    )
    weighted_clue_df.loc[:, enemy_cards] = (
        weighted_clue_df.loc[:, enemy_cards] * enemy_weight
    )
    weighted_clue_df.loc[:, ally_cards] = (
        weighted_clue_df.loc[:, ally_cards] * ally_weight
    )
    weighted_clue_df["weighted_score"] = weighted_clue_df.iloc[:, :25].sum(axis=1)
    weighted_clue_df["weighted_score"] = weighted_clue_df.weighted_score + (
        weighted_clue_df.actual_combo_length * risk_weight
    )
    best_clue = weighted_clue_df.iloc[weighted_clue_df["weighted_score"].argmax()]

    return best_clue, weighted_clue_df


def restrict_vocab_with_set(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        #         vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
    #             new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)


def create_amended_combos(df):
    #  Can be done with apply, doing it dirtily in the hope of a heavy refactor after the fact
    new_combos = []
    for idx, row in df.iterrows():
        intended_combo = idx[1]
        zero_values = row.loc[row == 0]
        new_combo = [c for c in intended_combo if c not in zero_values.index]
        new_combos.append(new_combo)
    return new_combos


def check_card_type(idx, spymaster_map, revealed):
    revealed_idxs = np.where(np.asarray(revealed) == True)[0]
    if idx not in revealed_idxs:
        return "hidden"
    for card_type, type_idxs in spymaster_map.items():
        if idx in type_idxs:
            return card_type
