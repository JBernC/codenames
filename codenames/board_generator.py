import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from codenames_board import CodenamesGame

parser = ArgumentParser(
    description='Generate datasets of codenames board for use in training')
parser.add_argument('--ds', type=int)
parser.add_argument('--f', type=str)


args = parser.parse_args()

board_df= pd.DataFrame(columns=['cards', 'blue_positions', 'red_positions', 'neutral_positions', 'assassin_positions', 'revealed', 'current_team'])
for i in tqdm(range(args.ds)):
    temp_game = CodenamesGame()
    temp_game = temp_game.get_game()
    board_df = board_df.append(temp_game, ignore_index=True)

board_df.to_hdf(f'../data/{args.f}.h5', 'data')   