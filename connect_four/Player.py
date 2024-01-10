import numpy as np
DEPTH_LIM=3
class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.opponent_number = 1
        if self.player_number == 1:
            self.opponent_number = 2
        if(self.player_number == 2):
            self.opponent_number = 1 
        self.nodes_explored = 0    

    def winning_state(self,board):
    # checking horizontal 'windows' of 4 for win
        COLS = board.shape[1]
        ROWS = board.shape[0]
        piece = self.player_number
        for c in range(COLS-3):
            for r in range(ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True

        # checking vertical 'windows' of 4 for win
        for c in range(COLS):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True

        # checking positively sloped diagonals for win
        for c in range(COLS-3):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True

        # checking negatively sloped diagonals for win
        for c in range(3,COLS):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c-1] == piece and board[r-2][c-2] == piece and board[r-3][c-3] == piece:
                    return True       

    def valid_cols_rows(self,board): # Returns the row and  col of next possible move
        row_col = []
        for col in range(board.shape[1]-1,0,-1):
            row = 5
            while board[row][col]!=0 and row>0:
                row = row - 1
            if board[row][col]==0:
                row_col.append((row,col))
        return row_col                


    def result(self,board,action,number): # Check result with a copy of the board
        b_copy = np.zeros([6,7]).astype(np.uint8)
        for row_index in range(len(board)):
            if row_index != action[0]:
                b_copy[row_index] = board[row_index]
            else:
                new_row = []
                for col_index in range(len(board[0])):
                    if col_index != action[1]:
                        new_row.append(board[row_index][col_index])
                    else:
                        new_row.append(number)
                b_copy[row_index] = new_row
        return b_copy    


    def min_value(self,board,alpha,beta,depth):
        #print("in min_value with board: \n"+str(board))
        self.nodes_explored = self.nodes_explored+1
        if self.winning_state(board) or depth>=DEPTH_LIM:
            ret = (self.evaluation_function(board,self.player_number)-self.evaluation_function(board,self.opponent_number),3)
            #print("min,")
            # for i in range(depth):
            #     print(" ,")
            #print(ret)
            return ret
        util_val = 1000000
        move = 0
        move_pair = (move,util_val)
        actions = self.valid_cols_rows(board)
        for action in actions:
            action_util_val = self.max_value(self.result(board,action,self.opponent_number),alpha,beta,depth+1)[0]
            if action_util_val < util_val :
                move = action[1]
            #     print("c,")
            # print(move)

            util_val=min(util_val,action_util_val) 
            if util_val<alpha:
                
                # print("min,")
                # for i in range(depth):
                #     print(" ,")
                # print(util_val, move, " from ",actions)
                return (util_val,move)
            beta = min(beta,util_val)
            move_pair = (util_val,move)
        
        # print("min,")
        # for i in range(depth):
        #     print(" ,")
        # print(move_pair)
        return move_pair    

    def max_value(self,board,alpha,beta,depth):
        #print("in max_value with board "+str(board))
        self.nodes_explored = self.nodes_explored+1
        if self.winning_state(board) or depth>=DEPTH_LIM:
            ret = (self.evaluation_function(board,self.player_number)-self.evaluation_function(board,self.opponent_number),4)
            # print("max,")
            # for i in range(depth):
            #     print(" ,")
            # print(ret)
            return ret 
        util_val = -1000000
        move = 0
        move_pair = (move,util_val)
        actions = self.valid_cols_rows(board)
        for action in actions: 
            # print(action)
            action_util_val = self.min_value(self.result(board,action,self.player_number),alpha,beta,depth+1)[0]
            if action_util_val > util_val:
                move = action[1]
            #     print("c,")
            # print(move)
            
            util_val = max(util_val,action_util_val) 
            
            if util_val > beta:
                
                # print("max,")
                # for i in range(depth):
                #     print(" ,")
                # print(util_val, move, " from ",actions)
                return (util_val,move)
            alpha = max(alpha,util_val)
            move_pair = (util_val,move)
        
        # print("max,")

        # for i in range(depth):
        #     print(" ,")
        # print(move_pair)
        return move_pair    

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm
        This will play against either itself or a human player
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them
        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = 0
        beta = 0
        depth = 0
        move = self.max_value(board,alpha,beta,depth)
        print("The number of nodes explored ", self.nodes_explored)
        return move[1]
        raise NotImplementedError('Whoops I don\'t know what to do')


    def probability(self,board,action,actions):
        return int(1/len(actions))    

    def exp_value(self,board,layer):
        util_val = 0
        actions = self.valid_cols_rows(board) 
        for action in actions:
            p = self.probability(board,action,actions)
            print(self.value(self.result(board,action,self.opponent_number),True,layer+1))
            util_val = util_val + p*self.value(self.result(board,action,self.opponent_number),True,layer+1)[0]
        return util_val    

    def value(self,board,isMax,layer):
        if self.winning_state(board) or layer >= 3:
            if isMax:
                return (self.evaluation_function(board,self.player_number)-self.evaluation_function(board,self.opponent_number),2)
            else:
                return self.evaluation_function(board,self.player_number)-self.evaluation_function(board,self.opponent_number)
        
        if isMax:
            return self.max_value_expectimax(board,layer)
        else:
            return self.exp_value(board,layer)   


    def max_value_expectimax(self,board,layer):
        util_val = -1000000
        actions = self.valid_cols_rows(board)
        move = 2
        for action in actions:
            
            if self.value(self.result(board,action,self.player_number),False,layer+1) > util_val:
                move = action[1]
            util_val = max(util_val,self.value(self.result(board,action,self.player_number),False,layer+1))
        return (util_val,move)         


    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        print("expectimax")
        return self.value(board,True,0)[1]
        raise NotImplementedError('Whoops I don\'t know what to do')        


    def evaluate_window(self, window, piece):
        if(piece ==1): opponent_piece=2
        else: opponent_piece =1

        # initial score of a window is 0
        score = 0

        # based on how many friendly pieces there are in the window, we increase the score
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        # or decrese it if the oponent has 3 in a row
        if window.count(opponent_piece) == 3 and window.count(0) == 1:
            score -= 4
        elif window.count(opponent_piece) == 2 and window.count(0) == 2:
            score -= 1

        return score 

    def evaluation_function(self, board, piece):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        score = 0
        #piece = self.player_number
        COLS = board.shape[1]
        ROWS = board.shape[0]
        center_array = [int(i) for i in list(board[:,COLS//2])]
        center_count = center_array.count(piece) # Count how many player coins are there
        score += center_count * 6 # multiply that number by 6

        # below we go over every single window in different directions and adding up their values to the score
        # score horizontal
        for r in range(ROWS):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(COLS - 3):
                window = row_array[c:c + 4] # check 4 element window in every row
                score += self.evaluate_window(window, piece)

        # score vertical
        for c in range(COLS):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(ROWS-3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece)

        # score positively sloped diagonals
        for r in range(3,ROWS):
            for c in range(COLS - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        # score negatively sloped diagonals
        for r in range(3,ROWS):
            for c in range(3,COLS):
                window = [board[r-i][c-i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score
        

class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
        