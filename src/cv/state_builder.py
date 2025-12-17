import numpy as np

class StateBuilder:
    def __init__(self, initial_stack=100.0):
        self.initial_stack = initial_stack
        
        # Card encoding maps (must match PokerEnv)
        self.suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
        self.ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}

    def build_observation(self, cv_state):
        """
        Converts raw CV state dict to Gym observation dict.
        cv_state structure:
        {
            "hand": ["Ah", "Kd"],
            "board": ["2s", "3d", "4c"],
            "pot": 15.5,
            "players": [
                {"id": 0, "status": "active", "stack": 100.0, "bet": 0.0},
                ...
            ]
        }
        """
        
        # 1. Hand
        hand_cards = [self._encode_card(c) for c in cv_state.get("hand", [])]
        while len(hand_cards) < 2:
            hand_cards.append(-1)
            
        # 2. Board
        board_cards = [self._encode_card(c) for c in cv_state.get("board", [])]
        while len(board_cards) < 5:
            board_cards.append(-1)
            
        # 3. Pot (Normalized)
        pot = cv_state.get("pot", 0.0) / self.initial_stack
        
        # 4. My Stack & Bet
        # Assuming Hero is always at a specific index or identified
        # For now, let's assume Hero is player index 4 (as per CV module comments)
        hero_idx = 4
        hero_data = cv_state["players"][hero_idx] if len(cv_state["players"]) > hero_idx else {}
        
        my_stack = hero_data.get("stack", 0.0) / self.initial_stack
        my_bet = hero_data.get("bet", 0.0) / self.initial_stack
        
        # 5. Players Info
        players_info = np.zeros((6, 4), dtype=np.float32)
        for i, p_data in enumerate(cv_state.get("players", [])):
            if i >= 6: break
            
            status = p_data.get("status", "active")
            is_folded = 1.0 if status == "folded" else 0.0
            is_active = 1.0 if status == "active" else 0.0
            
            stack = p_data.get("stack", 0.0) / self.initial_stack
            bet = p_data.get("bet", 0.0) / self.initial_stack
            
            players_info[i] = [is_active, stack, bet, is_folded]
            
        # 6. Position
        # In CV, we are always in the same seat physically.
        # But logically, position depends on Dealer Button.
        # We need Dealer Button detection to determine relative position.
        # For now, hardcode or use seat index.
        position = hero_idx
        
        # 7. Street
        # Derived from number of board cards
        num_board = len([c for c in cv_state.get("board", []) if c != "NoCard"])
        street = 0 # Preflop
        if num_board >= 3: street = 1 # Flop
        if num_board >= 4: street = 2 # Turn
        if num_board == 5: street = 3 # River
        
        # 8. Legal Actions
        # CV cannot know legal actions directly (logic engine needed).
        # We must infer or assume all actions legal unless folded.
        # Or we need a separate Logic Module that tracks game state.
        # For now, assume [Fold, Call, Raise] are all valid if active.
        legal_mask = np.array([1, 1, 1], dtype=np.int32)
        
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
        if not card_str or card_str == "NoCard": return -1
        try:
            # Assuming card_str is like 'Ah', 'Td'
            rank_char = card_str[0].upper()
            suit_char = card_str[1].upper()
            
            rank = self.ranks.get(rank_char, -1)
            suit = self.suits.get(suit_char, -1)
            
            if rank == -1 or suit == -1: return -1
            return rank * 4 + suit
        except:
            return -1
