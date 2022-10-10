import numpy as np

class QAgent:
    """
    Class holding a Q-learning agent for tic-tac-toe
    """

    def __init__(self, epsilon=0.99) -> None:
        self.q_table = {}
        self.epsiolon = epsilon

    def get_q_table_entry(self, state: np.ndarray) -> np.ndarray:
        """
        Create a new entry for the Q-table
        All nonzero entries are invalid actions so set them to NaN
        """
        key = state.tobytes()
        try:
            entry = self.q_table[key]
        except KeyError:
            entry = np.zeros_like(state, dtype=float)
            entry[state != 0] = np.nan
            self.q_table[key] = entry
        return entry

    def epsilon_greedy_action(self, state: np.ndarray) -> tuple[int, int]:
        """
        Get an action with the epsilon greedy method
        """
        if np.random.random() < self.epsiolon:
            # Choose a complelty random action
            return self.random_action(state)
        else:
            # Choose the action with the highest expected future reward
            return self.argmax_action(state)

    def random_action(self, state: np.ndarray) -> tuple[int, int]:
        """
        Return a random action
        """
        # Find all valid actions
        valid_actions, = np.flatnonzero(state == 0)
        # Select a random action
        i = np.random.randint(valid_actions.size)
        return np.unravel_index(valid_actions[i], state.shape)

    def argmax_action(self, state: np.ndarray) -> tuple[int, int]:
        """
        Return action with the bigest expected future reward
        """
        table = self.get_q_table_entry(state)
        return np.unravel_index(np.nanargmax(table), table.shape)


    
