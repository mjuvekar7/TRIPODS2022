import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from PIL import ImageFont


"""Markdown
# GRIDWORLD
    This class allows us to define a grid game and its basic characteristics such as the size of the board, 
    where the agent starts and currently is during an episode, and what its reward matrix and possible moves are

"""

class GridWorld:
    def __init__(self, board_dims, start_state, reward_matrix):
        self.board_dims = board_dims
        self.height, self.width = board_dims
        self.start_state = start_state
        self.agent_pos = start_state
        self.reward_matrix = reward_matrix
        self.moves = [(-1,0),(0,-1),(1,0),(0,1)] #up,left,down,right

    def draw(self):
        pass

    def update_state(self):
        pass

    def is_episode_terminal(self):
        pass

    def refresh_game(self):
        pass

    def get_moves(self, point):
        """
        point - the point our agent is in
        This function obtains the points of possible moves you can potentially make from the point provided,
        return being the ending position of the move

        """
        return [tuple(np.array(point)+np.array(x)) for x in self.moves]

    def get_reward(self,point):
        """
        point - a point in our grid
        This function returns the reward for our agent going to the state represented by point
        """
        return self.reward_matrix[point]
    
    def illegal_move(self,point):
        """
        point - a point that may or may not be in our grid
        This function returns true if the move we make takes us off the board and false if the move is legal
        """
        return True if ((point[0] < 0) or (point[0] == self.height) or (point[1] < 0) or (point[1] == self.width)) else False
        
    def print_board(self):
        print(self.reward_matrix)

"""Markdown
## Targeting Game
    The targeting game tasks consists of the agent starting in a different square from the "target" square. 
    The goal of the game is to have the agent reach the target square in as few moves as possible
"""

class Targeting_Game(GridWorld):
    def __init__(self, board_dims, start_state, terminal_state, reward_matrix):
        GridWorld.__init__(self,board_dims, start_state, reward_matrix)
        self.terminal_state = terminal_state
        
    def draw(self):
        """
        agent_pos - tuple point representing agent's current position on game board 
        remaining_prizes - list of tuples representing remaining prizes and their positions on game board
        This function draws our environment
        """

        image = Image.new("RGB", (501, 501), "black")
        draw = ImageDraw.Draw(image)

        w = 500/(self.width)
        h = 500/(self.height)
        color = "white"

        #Draw Grid and Start/Stop Squares
        for i in range(self.height):
            for j in range(self.width):
                if(i == self.start_state[0] and j == self.start_state[1]):
                    color = "blue"
                elif(i == self.terminal_state[0] and j == self.terminal_state[1]):
                    color = "red"
                draw.rectangle(((0+j*w, 0+i*h),(w+j*w, h+i*h)), outline = "black", fill = color)
                color ="white"

        #Draw Agent
        draw.ellipse((self.agent_pos[1]*w + w/4, self.agent_pos[0]*h + h/4, 3*w/4 + self.agent_pos[1]*w, 3*h/4 + self.agent_pos[0]*h), fill="black")

        display(image)

    def update_state(self, new_pos, illegal):
        """
        new_pos - a point in the game grid that the agent has moved to
        This function updates the agent position for the GameGrid class variable.
        """
        if not illegal:
            self.agent_pos = new_pos

    def is_episode_terminal(self):
        """
        This function returns a boolean based on if the Game's current episode is finished.
        """
        return True if (self.agent_pos == self.terminal_state) else False

    def refresh_game(self):
        """
        This function refreshes the important features of the game that might have changed within an episode
        """
        self.agent_pos = self.start_state

"""Markdown
## Collection Game
    The collection game tasks consists of the agent starting in a different square from a set of prize squares. 
    The goal of the game is to have the agent collect each prize in the prize squares in as few moves as possible 
    (prizes being removed upon collection)
"""

class Collection_Game(GridWorld):
    def __init__(self, board_dims, start_state, prize_states, reward_matrix, prize_value=0):
        GridWorld.__init__(self,board_dims, start_state, reward_matrix)
        self.prize_states = prize_states
        self.remaining_prize_states = list(prize_states)
        self.prize_value = prize_value
        for prize_state in prize_states:
            self.reward_matrix[prize_state] = prize_value
    
    def draw(self):
        """
        agent_pos - tuple point representing agent's current position on game board 
        remaining_prizes - list of tuples representing remaining prizes and their positions on game board
        This function draws our environment
        """

        image = Image.new("RGB", (501, 501), "black")
        draw = ImageDraw.Draw(image)

        w = 500/(self.width)
        h = 500/(self.height)
        color = "white"

        #Draw Grid and Start/Stop Squares
        for i in range(self.height):
            for j in range(self.width):
                if(i == self.start_state[0] and j == self.start_state[1]):
                    color = "blue"
                draw.rectangle(((0+j*w, 0+i*h),(w+j*w, h+i*h)), outline = "black", fill = color)
                color ="white"

        #Draw Agent
        draw.ellipse((self.agent_pos[1]*w + w/4, self.agent_pos[0]*h + h/4, 3*w/4 + self.agent_pos[1]*w, 3*h/4 + self.agent_pos[0]*h), fill="black")

        #Draw Prizes
        if len(self.remaining_prize_states) > 1:
            for x in self.remaining_prize_states:
                draw.rectangle(((x[1]*w + w/4, x[0]*h + h/4), (3*w/4+x[1]*w, 3*h/4+x[0]*h)), outline = "black", fill = "yellow")
        elif len(self.remaining_prize_states) == 1:
            remaining_prize = self.remaining_prize_states[0]
            draw.rectangle(((remaining_prize[1]*w + w/4, remaining_prize[0]*h + h/4), (3*w/4+remaining_prize[1]*w, 3*h/4+remaining_prize[0]*h)), outline = "black", fill = "yellow")
      
        display(image)

    def update_state(self, new_pos, illegal):
        """
        new_pos - a point in the game grid that the agent has moved to
        This function updates the agent position for the GameGrid class variable.
        """
        if not illegal:
            self.agent_pos = new_pos

        if self.agent_pos in self.remaining_prize_states:
            self.remove_prize(self.agent_pos)

    def remove_prize(self, prize_point):
        """
        prize_point - a point in the game grid that contained a prize
        This function removes the prize at the prize_point supplied from the remaining prizes. This function also
            updates the reward matrix accordingly.
        """
        #remove prize from remaining prizes
        self.remaining_prize_states.remove(prize_point)
        #adjust reward matrix to account for no prize at this prize_point for the rest of the episode
        self.reward_matrix[prize_point] = self.reward_matrix[self.start_state]
        #if there remains one prize, set that to be the terminal state for the episode
        if len(self.remaining_prize_states) == 1:
            self.terminal_state = self.remaining_prize_states[0]
        
    def is_episode_terminal(self):
        return True if len(self.remaining_prize_states) == 0 else False

    def refresh_game(self):
        """
        This function refreshes the game's agent position, the remaining prizes, the reward matrix, and terminal 
            state. This is used between each episode
        """
        self.agent_pos = self.start_state
        self.remaining_prize_states = list(self.prize_states)
        self.terminal_state = None
        
        for prize_state in self.prize_states:
            self.reward_matrix[prize_state] = self.prize_value

"""Markdown
## FindMax Game
    The FindMax game consists of the agent starting in a square with every other square intialized with some reward. 
    The goal of the game is to have the agent find the highest reward square. Ignoring rewards, this is the exact
    same game as our Targeting game except this game has a more complex reward landscape for an agent to manuever.
    There is a chance that if the agent is too simple, that they'll get stuck and ignore finding the maximum value
    because there isn't enough reward to finding it vs not. 
    
    TODO:
    Therefore, we should add a negative reward for not finishing the game on the right square. We could make this a
    unanimous feature for our games and setting step limits for all games.
"""

class FindMax_Game(GridWorld):
    def __init__(self, board_dims, start_state, reward_matrix):
        GridWorld.__init__(self, board_dims, start_state, reward_matrix)
        # the terminal state is where the max value locates
        self.max_state = np.unravel_index(np.argmax(reward_matrix, axis=None), board_dims)
        # initial sum of rewards is the reward value at the start state
        self.total_rewards = reward_matrix[start_state]
        # the number of steps
        self.n_steps = 0
        # the step limit is the Manhattan distance between the start state and the terminal state
        self.step_limit = sum(abs(np.array(self.max_state)-np.array(start_state)))

    def draw(self):
        image = Image.new("RGBA", (501, 501), (255, 255, 255, 255)) # white
        draw = ImageDraw.Draw(image)

        w = 500/(self.width)
        h = 500/(self.height)
        color = (255, 255, 255, 0) # white, transparent

        # use !fc-list or !fc-list | grep "" to get the path of the font-type on colab
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40) 
        
        #Draw Grid
        for i in range(self.height):
            for j in range(self.width):
                if(i == self.start_state[0] and j == self.start_state[1]):
                    color = (0, 0, 255, 255)  # blue for the starting point
                if(i == self.max_state[0] and j == self.max_state[1]):
                    color = (255, 0, 0, 255)  # red for the max
                draw.rectangle(((0+j*w, 0+i*h),(w+j*w, h+i*h)), outline = "black", fill = color)
                text_h = font.getsize(str(self.reward_matrix[(i,j)]))[1]
                text_w = font.getsize(str(self.reward_matrix[(i,j)]))[0]
                draw.text((j*w + w/2 - text_w/2, i*h + h/2 - text_h/2), 
                          str(self.reward_matrix[(i,j)]), font=font, fill=(0, 0, 0, 255))
                color = (255, 255, 255, 0)

        #Draw Agent
        agent_layer = Image.new('RGBA', (501, 501), (255, 255, 255, 0))
        draw2 = ImageDraw.Draw(agent_layer)
        draw2.ellipse((self.agent_pos[1]*w + w/4, self.agent_pos[0]*h + h/4, 3*w/4 + self.agent_pos[1]*w, 3*h/4 + self.agent_pos[0]*h), 
                     fill=(255, 0, 255, 128)) 
      
        out = Image.alpha_composite(image, agent_layer)
        display(out)
      
    def update_state(self, new_pos, illegal, illegal_reward=-100):
        """
        new_pos - a point in the game grid that the agent has moved to
        a function to update the position of agent, number of steps, and sum of rewards,
        """
        if not illegal:
            self.agent_pos = new_pos
            self.total_rewards += self.reward_matrix[self.agent_pos]
        else:
            self.total_rewards += illegal_reward
        self.n_steps += 1 # once the agent moves, the number of steps taken (var n_step) +1

    def is_episode_terminal(self):
      return True if (self.agent_pos == self.max_state or self.n_steps == self.step_limit) else False

    def refresh_game(self):
        self.agent_pos = self.start_state
        self.total_rewards = self.reward_matrix[self.start_state] 
        self.n_steps = 0

"""Markdown
## MaxPath Game
    The MaxPath game consists of the agent starting in a square with every other square intialized with some reward. 
    The goal of the game is to have the agent achieve the highest reward possible in a set amount of moves.
"""

class MaxPath_Game(GridWorld):
    def __init__(self, board_dims, start_state, reward_matrix, n_allowed_moves):
        GridWorld.__init__(self, board_dims, start_state, reward_matrix)
        # initial sum of rewards is the reward value at the start state
        self.total_rewards = reward_matrix[start_state] 
        self.n_steps = 0
        self.step_limit = n_allowed_moves

    def draw(self):
        image = Image.new("RGBA", (501, 501), (255, 255, 255, 255)) # white
        draw = ImageDraw.Draw(image)

        w = 500/(self.width)
        h = 500/(self.height)
        color = (255, 255, 255, 0) # white, transparent

        # use !fc-list or !fc-list | grep "" to get the path of the font-type on colab
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 40) 
        
        
        #Draw Grid 
        for i in range(self.height):
            for j in range(self.width):
                if(i == self.start_state[0] and j == self.start_state[1]):
                    color = (0, 0, 255, 255)  # blue for the starting point
                if(i == self.max_state[0] and j == self.max_state[1]):
                    color = (255, 0, 0, 255)  # red for the max
                draw.rectangle(((0+j*w, 0+i*h),(w+j*w, h+i*h)), outline = "black", fill = color)
                text_h = font.getsize(str(self.reward_matrix[(i,j)]))[1]
                text_w = font.getsize(str(self.reward_matrix[(i,j)]))[0]
                draw.text((j*w + w/2 - text_w/2, i*h + h/2 - text_h/2), 
                          str(self.reward_matrix[(i,j)]), font=font, fill=(0, 0, 0, 255))
                color = (255, 255, 255, 0)

        #Draw Agent
        agent_layer = Image.new('RGBA', (501, 501), (255, 255, 255, 0))
        draw2 = ImageDraw.Draw(agent_layer)
        draw2.ellipse((self.agent_pos[1]*w + w/4, self.agent_pos[0]*h + h/4, 3*w/4 + self.agent_pos[1]*w, 3*h/4 + self.agent_pos[0]*h), 
                     fill=(255, 0, 255, 128)) 
      
        out = Image.alpha_composite(image, agent_layer)
        display(out)
      
    def update_state(self, new_pos, illegal, illegal_reward=-100):
        """
        new_pos - a point in the game grid that the agent has moved to
        a function to update the position of agent, number of steps, and sum of rewards,
        """
        if not illegal:
            self.agent_pos = new_pos
            self.total_rewards += self.reward_matrix[self.agent_pos]
        else:
            self.total_rewards += illegal_reward
        self.n_steps += 1 # once the agent moves, the number of steps taken (var n_step) +1

    def is_episode_terminal(self):
        return True if self.n_steps == self.step_limit else False

    def refresh_game(self):
        self.agent_pos = self.start_state
        self.total_rewards = self.reward_matrix[self.start_state] 
        self.n_steps = 0

"""
## Example
"""
def main():
    """
    Define the grid game's reward matrix and initialize game
    """
    reward_matrix = np.ones((3,3))*-1
    reward_matrix[(2,2)] = 0
    print(reward_matrix)
    target_game = Targeting_Game(board_dims=(3,3), start_state=(0,0), terminal_state=(2,2), reward_matrix=reward_matrix)
    target_game.draw()

if __name__ == "__main__":
    main()