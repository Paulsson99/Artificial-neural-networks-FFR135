import os
import numpy as np
from tqdm import trange

from game import TicTacToe
from player import QAgent


GameHistory = tuple[tuple[int, int], np.ndarray]


def play_game(agent1: QAgent, agent2: QAgent, epsilon: float, verbose=False) -> tuple[float, GameHistory]:
    """
    Play a game between two agents with the epsilon greeady approch
    """
    game = TicTacToe()
    game_history = []

    if verbose:
        print(game)
        print()

    while True:
        # Let player 1 move
        action1 = agent1.epsilon_greedy_action(state=game.board, epsilon=epsilon)

        # Add the state to the history
        game_history.append((action1, game.board.copy()))

        game.play(move=action1, player=1)

        if verbose:
            print(game)
            print()

        if game.game_over():
            break

        # Let player 2 move
        action2 = agent2.epsilon_greedy_action(state=game.board, epsilon=epsilon)

        # Add the state to the history
        game_history.append((action2, game.board.copy()))

        game.play(move=action2, player=-1)

        if verbose:
            print(game)
            print()
        
        if game.game_over():
            break

    reward = game.winner()
    return reward, game_history


def update_q_tables(agent: QAgent, game_history: list[GameHistory], reward: float, lr: float) -> None:
    """
    Update a q table
    """
    # Set the end reward
    action, state = game_history[-1]
    agent.set_reward(state, action, reward)

    expected_reward = reward
    for action, state in reversed(game_history[:-1]):
        expected_reward = agent.update_reward(state, action, expected_reward, lr)


def train(epochs: int, lr: float, agent1: QAgent = None, agent2: QAgent = None) -> tuple[QAgent, QAgent]:
    """
    Train the agents
    """
    agent1 = agent1 or QAgent()
    agent2 = agent2 or QAgent()

    epsilon = 1

    for epoch in trange(epochs):
        reward, game_history = play_game(agent1, agent2, epsilon)
        # Update player 1
        update_q_tables(agent1, game_history[::2], reward, lr)
        # Update player 2
        update_q_tables(agent2, game_history[1::2], -reward, lr)

        epsilon *= 0.9999

    return agent1, agent2


def test_agents(agent1: QAgent, agent2: QAgent, games: int) -> None:
    game_stats = {1: 0, 0: 0, -1: 0}
    for game in trange(games):
        reward, _ = play_game(agent1, agent2, epsilon=0)
        game_stats[reward] += 1
    
    print(f"Player 1 won {game_stats[1]} games.")
    print(f"Player 2 won {game_stats[-1]} games.")
    print(f"{game_stats[0]} games resulted in a draw.")


def main():
    agent1, agent2 = train(epochs=100000, lr=0.1)
    test_agents(agent1, agent2, 1000)

    agent1.save_q_table('player1.csv')
    agent2.save_q_table('player2.csv')
    # play_game(agent1, agent2, 0, verbose=True)


if __name__ == '__main__':
    # To save the data correctly
    os.chdir(os.path.dirname(__file__))

    main()