import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('no-limit-holdem', config={'game_num_players': 6})
state, player_id = env.reset()

print("Legal Actions:", state['legal_actions'])
print("Raw Legal Actions:", state['raw_legal_actions'])
