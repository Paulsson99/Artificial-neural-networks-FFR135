import numpy as np


class QAgent:
    """
    Class holding a Q-learning agent for tic-tac-toe
    """

    def __init__(self) -> None:
        self.q_table = {}

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

    def epsilon_greedy_action(self, state: np.ndarray, epsilon: float) -> tuple[int, int]:
        """
        Get an action with the epsilon greedy method
        """
        if np.random.random() < epsilon:
            # Choose a complelty random action
            return self.random_action(state)
        else:
            # Choose the action with the highest expected future reward
            return self.argmax_action(state)

    def random_action(self, state: np.ndarray) -> tuple[int, int]:
        """
        Return a random action
        """
        # Add the table
        self.get_q_table_entry(state)
        # Find all valid actions
        valid_actions = np.flatnonzero(state == 0)
        # Select a random action
        i = np.random.randint(valid_actions.size)
        return np.unravel_index(valid_actions[i], state.shape)

    def argmax_action(self, state: np.ndarray) -> tuple[int, int]:
        """
        Return action with the bigest expected future reward
        """
        table = self.get_q_table_entry(state)
        return np.unravel_index(np.nanargmax(table), table.shape)

    def set_reward(self, state: np.ndarray, action: tuple[int, int], reward: float) -> None:
        """
        Set the reward for an entry in the Q-table
        """
        key = state.tobytes()
        self.q_table[key][action] = reward

    def update_reward(self, state: np.ndarray, action: tuple[int, int], expected_reward: float, lr: float) -> float:
        """
        Upade the reward in an entry in the Q-table
        Return the new reward
        """
        key = state.tobytes()
        dQ = expected_reward - self.q_table[key][action]
        self.q_table[key][action] += lr * dQ
        return self.q_table[key][action]

    def save_q_table(self, out_file: str) -> None:
        """
        Save the Q-table
        """
        states = np.empty((3, 3 * len(self.q_table)))
        q_values = np.empty((3, 3 * len(self.q_table)))
        for i, (key, q_value) in enumerate(self.q_table.items()):
            # Convert the hashkey back to a state
            state = np.frombuffer(key, dtype=int).reshape((3, 3))
            
            states[:, 3 * i: 3 * (i + 1)] = state
            q_values[:, 3 * i: 3 * (i + 1)] = q_value

        # Save the states
        np.savetxt(out_file, states, delimiter=',', fmt='%d')

        # Save the Q-values
        with open(out_file, 'ab') as f:
            np.savetxt(f, q_values, delimiter=',')
        

            









    
