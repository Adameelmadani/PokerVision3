import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rlcard
from rlcard.agents import RandomAgent

class PokerEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Wraps rlcard Texas Hold'em to provide specific observation space for CV integration.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players=6):
        super(PokerEnv, self).__init__()
        self.num_players = num_players
        
        # Create rlcard environment
        self.game = rlcard.make('no-limit-holdem', config={'game_num_players': num_players})
        
        # Define Action Space
        # 0: Fold
        # 1: Check/Call
        # 2: Raise (Half Pot - simplified for now)
        # 3: Raise (Pot - simplified for now)
        # 4: All-in
        # For now, let's stick to the user's request: Fold, Check/Call, Raise (fixed)
        # User said: 0=Fold, 1=Check/Call, 2=Raise (fixed size)
        self.action_space = spaces.Discrete(3)

        # Define Observation Space
        # We use a Dict space to match the structure described
        # Cards are encoded as integers 0-51 (or similar), -1 for empty
        
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(low=-1, high=51, shape=(2,), dtype=np.int32),
            "board": spaces.Box(low=-1, high=51, shape=(5,), dtype=np.int32),
            "pot": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "my_stack": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "my_bet": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # Players info: 6 players * 4 features (active, stack, bet, folded)
            "players": spaces.Box(low=0, high=1, shape=(6, 4), dtype=np.float32),
            "position": spaces.Box(low=0, high=5, shape=(1,), dtype=np.int32),
            # Street: 0: Preflop, 1: Flop, 2: Turn, 3: River
            "street": spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),
            # Legal actions mask: [fold, call, raise]
            "legal_actions": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32),
        })

        self.initial_stack = 100.0 # Default in rlcard usually, need to check config
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.player_id = self.game.reset()
        return self._get_observation(self.state), {}

    def step(self, action):
        # Map gym action to rlcard action
        # rlcard actions: fold, check, call, raise, all-in
        # We need to map our 0, 1, 2 to valid rlcard actions
        
        legal_actions = self.state['legal_actions']
        # legal_actions is OrderedDict {action_id: None}
        
        mapped_action = self._map_action(action, legal_actions)
        
        self.state, self.player_id = self.game.step(mapped_action)
        
        # Reward: 
        # In rlcard, reward is usually given at the end of the game.
        # If game is not done, reward is 0.
        
        done = self.game.is_over()
        reward = 0
        if done:
            # rlcard returns payoffs for all players
            payoffs = self.game.get_payoffs()
            reward = payoffs[self.player_id]
            
        obs = self._get_observation(self.state)
        
        return obs, reward, done, False, {}

    def _map_action(self, action, legal_actions):
        # 0: Fold
        # 1: Check/Call
        # 2: Raise
        
        legal_ids = list(legal_actions.keys())
        
        # Logic to choose best matching legal action
        if action == 0:
            if 0 in legal_ids: return 0 # Fold
            if 1 in legal_ids: return 1 # Check if can't fold (shouldn't happen usually)
            
        if action == 1:
            if 1 in legal_ids: return 1 # Check/Call
            if 0 in legal_ids: return 0 # Fold if can't call
            
        if action == 2:
            # Try various raise sizes
            if 2 in legal_ids: return 2 # Raise Half Pot
            if 3 in legal_ids: return 3 # Raise Pot
            if 4 in legal_ids: return 4 # All In
            if 1 in legal_ids: return 1 # Call if can't raise
            
        return legal_ids[0] # Fallback

    def _get_observation(self, state):
        raw_obs = state['raw_obs']
        
        # 1. Hand
        hand_cards = [self._encode_card(c) for c in raw_obs['hand']]
        # Pad to 2 cards if needed (should be 2 unless empty?)
        while len(hand_cards) < 2:
            hand_cards.append(-1)
            
        # 2. Board
        board_cards = [self._encode_card(c) for c in raw_obs['public_cards']]
        while len(board_cards) < 5:
            board_cards.append(-1)
            
        # 3. Pot
        pot = raw_obs['pot'] / self.initial_stack
        
        # 4. My Stack & Bet
        my_stack = raw_obs['my_chips'] / self.initial_stack
        # raw_obs doesn't explicitly give "my_bet" for the current round easily, 
        # but we can infer it or just use 0 for now if not available.
        # Actually, 'all_chips' might be current stack, so initial - current = bet?
        # Let's assume 'all_chips' are current stacks.
        # For 'my_bet', we might need to track it manually or look at action history.
        # For now, let's set my_bet to 0.0 as a placeholder or try to derive it.
        my_bet = 0.0 
        
        # 5. Players Info
        # players: 6x4 [active, stack, bet, folded]
        players_info = np.zeros((6, 4), dtype=np.float32)
        all_chips = raw_obs['all_chips']
        
        # Determine who folded from action_record
        folded_players = set()
        # action_record is list of (player_id, action_str)
        # We need access to the game history, but state['action_record'] might be just for this round?
        # rlcard state['action_record'] accumulates.
        if 'action_record' in state:
            for pid, action in state['action_record']:
                if action == 'fold':
                    folded_players.add(pid)
        
        for i in range(self.num_players):
            # Active: if not folded and has chips (or is all-in)
            # Simplified: Active if not folded.
            is_folded = 1.0 if i in folded_players else 0.0
            stack = all_chips[i] / self.initial_stack
            
            # Bet: again, hard to get exact current round bet from raw_obs without tracking.
            # We'll leave bet as 0 for now.
            bet = 0.0
            
            is_active = 1.0 if not is_folded else 0.0
            
            players_info[i] = [is_active, stack, bet, is_folded]
            
        # 6. Position
        # current_player is the agent's ID.
        # But we want "position" relative to button? 
        # rlcard doesn't explicitly give button position in raw_obs easily.
        # We'll just use player_id as position for now.
        position = self.player_id
        
        # 7. Street
        # stage: 0=Preflop, 1=Flop, 2=Turn, 3=River
        street_map = {'PREFLOP': 0, 'FLOP': 1, 'TURN': 2, 'RIVER': 3, 'END_HIDDEN': 4, 'SHOWDOWN': 5}
        # rlcard raw_obs['stage'] is an Enum or string?
        # Let's check inspection output... it said 'stage'.
        # Usually it's a Stage enum. We'll cast to int if possible or map string.
        # Based on inspection, it might be an object. Let's assume int for now or handle exception.
        street = 0
        try:
            street = int(raw_obs['stage'])
        except:
            # If it's an Enum, .value might work, or string map
            s_str = str(raw_obs['stage']).split('.')[-1] # e.g. Stage.PREFLOP -> PREFLOP
            street = street_map.get(s_str, 0)

        # 8. Legal Actions
        # Mask: [fold, call, raise]
        legal_mask = np.zeros(3, dtype=np.int32)
        legal_ids = list(state['legal_actions'].keys())
        
        if 0 in legal_ids: legal_mask[0] = 1
        if 1 in legal_ids: legal_mask[1] = 1
        if 2 in legal_ids or 3 in legal_ids or 4 in legal_ids: legal_mask[2] = 1
        
        return {
            "hand": np.array(hand_cards, dtype=np.int32),
            "board": np.array(board_cards, dtype=np.int32),
            "pot": np.array([pot], dtype=np.float32),
            "my_stack": np.array([my_stack], dtype=np.float32),
            "my_bet": np.array([my_bet], dtype=np.float32),
            "players": players_info,
            "position": np.array([position], dtype=np.int32),
            "street": np.array([street], dtype=np.int32),
            "legal_actions": legal_mask
        }

    def _encode_card(self, card_str):
        # card_str: e.g. 'DT' (Diamond Ten)
        # Suits: S=0, H=1, D=2, C=3
        # Ranks: 2=0 ... A=12
        if card_str == 'NoCard': return -1
        
        suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
        ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        
        try:
            suit = suits[card_str[0]]
            rank = ranks[card_str[1]]
            return rank * 4 + suit
        except:
            return -1
