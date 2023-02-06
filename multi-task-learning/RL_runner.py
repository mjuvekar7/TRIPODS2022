import numpy as np
import random
import itertools
from tabular_agents import *
from games import *

class RL_Runner:
    """
    This class takes a game and agent object and allows the agent to play that game. At the moment, we only have grid games
        but hopefully in the future, due to the generalized nature of this code, we can use it to test different types of
        games and agents together
    """
    def __init__(self, Game, Agent):
        self.Game = Game
        self.Agent = Agent
        self.Agent.set_game(self.Game, is_board_game=True)

    def play_episode(self):
        """
        This function enables the agent to play an episode of the game. It has checks for illegal moves, performs
            alpha and epsilon decay after the episode completes, and records total reward of episode
        """
        t = 0
        episode_reward = 0
        #while agent is not in a terminal state, episode is still being played
        while not self.Game.is_episode_terminal():
            t += 1

            """
            Record current state S, get and take action A, record S'
            """
            current_state = self.Game.agent_pos
            new_action = self.Agent.get_action(self.Game.agent_pos) #get a possible action (up, left, down, and right for grid)
            new_state = tuple(np.array(self.Game.agent_pos) + np.array(new_action))

            """
            Check legality of action and get reward R, agent moves to S' in game object
            """
            #if move is illegal (going off the board), set reward to very bad
            if self.Game.illegal_move(new_state):
                reward = -100
                self.Game.update_state(new_state, illegal=True)
            else: #else the selected move is legal and we should get reward R for agent going to state S'
                reward = self.Game.get_reward(new_state)
                self.Game.update_state(new_state, illegal=False)
            
            episode_reward += reward
    
            self.Agent.update_model(False, current_state, new_action, reward)
            
        """
        Updates after episode is complete
        """
        self.Agent.update_model(episode_done=True)
        self.Agent.QTable.alpha_decay()
        self.Agent.QTable.epsilon_decay()
        return episode_reward
                
    
    def play_game(self, n_episodes, output=False):
        """
        n_episodes - an integer that corresponds to the number of times your agent plays the game
        This function has your agent play the game and learn how to play it. It records and prints episodic rewards
            and draws the grid game
        """
        player_scores = []

        self.Game.draw()
        
        for i in range(n_episodes):
            """
            Play and refresh an episode of the game
            """
            episode_reward = self.play_episode()
            player_scores.append(episode_reward)
            self.Game.refresh_game()
            
            print("Reward for Episode: ",i," -> ",episode_reward)

        print("Player scores for every episode: ",player_scores)
        self.Game.refresh_game()
        self.Game.draw()



def main():
    """
    Define the grid game's reward matrix
    """
    reward_matrix = np.ones((3,3))*-1
    reward_matrix[(2,2)] = 0
    print(reward_matrix)

    """
    Initialize the game and agent
    """
    target_game = Targeting_Game( (3,3), (0,0), (2,2), reward_matrix)
    mc_agent = OnPolicy_MCAgent(policy_update='softmax',alpha=0.2, epsilon=0.5, discount=0.5, \
                                        alpha_decay_rate = 0.9, epsilon_decay_rate=0.9)

    """
    Configure (Agent,Game) environment through runner class and play the game
    """
    env = RL_Runner(target_game, mc_agent)
    env.play_game(1000)
    mc_agent.print_models()


if __name__ == "__main__":
    main()
