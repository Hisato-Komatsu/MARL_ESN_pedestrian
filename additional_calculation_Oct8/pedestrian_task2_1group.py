"""
Source code for task.II
2024/2/15 modified the calculation method of matrix A
2024/9/28 additional research for the case that two groups share the parameters of neural network
"""
import numpy as np
import time
import os
import scipy
import copy
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix, csc_matrix
from collections import namedtuple
from collections import deque

start_time = time.time()
np.set_printoptions(precision=3)

seed = 1
np.random.seed(seed)

currentdir = os.getcwd()
global condition_dict
condition_dict = {"vacant":0, "wall":1, "agent":2}

class hyperparameters:
   def __init__(self):
      self.t_max = 500
      self.t_observe = 100
      self.ep_termination = 250
      self.ep_observe = 150
      self.road_width = 8
      self.width = 20
      self.height = 20
      self.n_agent = 48 #the number of agents
      self.agent_eyesight_whole = 11 #odd number

      self.leaking_rate = 0.8
      self.gamma = 0.95
      self.reservoir_size = 1024
      self.sparsity=0.9 #p^{in} _{s3}, p^{in} _{sb}, and p^{res} _s
      self.central_sparsity = 0.8 #p^{in} _{s2}
      self.core_sparsity = 0.6 #p^{in} _{s1}
      self.spectral_radius=0.95 
      self.beta = 0.0001 #Ridge term
      self.epsilon = 1.0 #epsilon-greedy
      self.epsilon_decay = 0.95
      self.epsilon_min = 0.02
      self.decay_ratio = 0.95 #decay of memory matrices per one update of W_out

      self.n_group = 2 #number of groups of agents

      self.sigma_W_in_a = 2.0 #stddev of W_in_a

def Adjacent(pos1,pos2):
   if abs(pos1[0]-pos2[0]) == 1 and abs(pos1[1]-pos2[1]) == 0:
      return True
   if abs(pos1[0]-pos2[0]) == 0 and abs(pos1[1]-pos2[1]) == 1:
      return True
   else:
      return False

class Env:
   def __init__(self, hyperparameters, render_mode='off', experience_sharing='on'):
      #render_mode : 'off', 'CUI', or 'rgb_array'
      self.render_mode = render_mode
      # experience_sharing : 'on' or 'off' ... It means whether parameter sharing is adopted or not.
      self.experience_sharing = experience_sharing
      if self.render_mode == 'rgb_array' :
         self.Image_list = []

      self.agents = []
      self.walls = []
      self.hyperparameters = hyperparameters
      self.road_width = self.hyperparameters.road_width
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.area = self.width*self.height
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.view_3ch = np.zeros((self.width,self.height,3))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.pos_to_object = [[None for i in range(self.height)] for j in range(self.width)]
      self.n_agent = self.hyperparameters.n_agent
      self.n_group = self.hyperparameters.n_group
      self.n_agent_1group = self.n_agent//self.n_group
      self.done = False
      self.epsilon_min = self.hyperparameters.epsilon_min

      self.build_wall()

      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         if i_agent == 0:
            new_agent = Agent(pos_temp_agent[0], pos_temp_agent[1], self.condition, kind='agent', hyperparameters=self.hyperparameters)
         else:
            new_agent = copy.deepcopy(self.agents[0])
            new_agent.pos = pos_temp_agent
         if i_agent <= self.n_agent_1group - 1 :
            new_agent.group = 0
            self.view_3ch[new_agent.pos[0]][new_agent.pos[1]][0] = new_agent.color
         else:
            new_agent.group = 1
            self.view_3ch[new_agent.pos[0]][new_agent.pos[1]][2] = new_agent.color
         self.agents.append(new_agent)
         self.pos_to_object[new_agent.pos[0]][new_agent.pos[1]] = new_agent
         self.condition[new_agent.pos[0]][new_agent.pos[1]] = condition_dict["agent"]
         self.view[new_agent.pos[0]][new_agent.pos[1]][0] = self.agents[i_agent].color

   def agent_position(self, i_agent):
      if i_agent <= self.n_agent_1group - 1 :
         return np.array([0+(i_agent%2)+2*(i_agent//self.road_width),7+(i_agent%self.road_width)])
      elif self.n_agent_1group <= i_agent <= self.n_agent - 1 :
         return np.array([19-(i_agent%2)-2*((i_agent-self.n_agent_1group)//self.road_width),7+((i_agent-self.n_agent_1group)%self.road_width)])

   def build_wall(self):
      if self.walls: 
         for w1 in self.walls:
            self.pos_to_object[w1.pos[0]][w1.pos[1]] = w1
            self.view[w1.pos[0]][w1.pos[1]][1] = w1.color
            self.condition[w1.pos[0]][w1.pos[1]] = condition_dict["wall"]
            self.view_3ch[w1.pos[0]][w1.pos[1]][1] = w1.color
      else:
         for x in range(self.width):
            for y in range(self.height):
               if y == 5 or y == 6:
                  self.set_wall(x,y)
               elif y == 7+self.road_width or y == 8+self.road_width:
                  self.set_wall(x,y)


   def set_wall(self,x,y):
      new_wall = Object(x,y, self.condition, kind=("wall"), hyperparameters=self.hyperparameters)
      self.walls.append(new_wall)
      self.pos_to_object[x][y] = new_wall
      self.view[x][y][1] = new_wall.color
      self.view_3ch[x][y][1] = new_wall.color

   def reset(self):
      #reset of environment with keeping the result of training
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.view_3ch = np.zeros((self.width,self.height,3))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.pos_to_object = [[None for i in range(self.height)] for j in range(self.width)]
      self.done = False
      self.build_wall()
      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         self.agents[i_agent].X_esn = np.zeros(self.agents[i_agent].reservoir_size)
         self.agents[i_agent].pos = pos_temp_agent
         self.agents[i_agent].total_reward = 0.0
         self.pos_to_object[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.agents[i_agent]
         self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = condition_dict["agent"]
         self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
         if self.agents[i_agent].group == 0 :
            self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
         elif self.agents[i_agent].group == 1 :
            self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][2] = self.agents[i_agent].color

   def _step(self, t):
      #agents' action
      for i_agent in range(self.n_agent):
         action=[0,0]
         #self.agents[i_agent]._key_step(self.pos_to_object)
         
         if t >= 1:
            self.agents[i_agent].reward_past = self.agents[i_agent].reward
            self.agents[i_agent].action_past = action
            self.agents[i_agent].Xi_temp_part0 = copy.deepcopy(self.agents[i_agent].Xi_temp_part1)

         delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
         dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
         state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
         self.agents[i_agent].action, self.agents[i_agent].Xi_temp_part1, self.agents[i_agent].Q_temp = self.agents[i_agent].choose_action(state)
         self.agents[i_agent]._step(self.agents[i_agent].action, self.pos_to_object)
         self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] += 1            
         
      #update of the positions of agents
      for i_agent in range(self.n_agent):
         if self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == condition_dict["vacant"] and self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == 1 :
            self.agents[i_agent].move_permission = 'yes'

      for i_agent in range(self.n_agent):
         #update of the position of agents
         if self.agents[i_agent].move_permission == 'yes' :
            self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = condition_dict["vacant"]
            self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] = condition_dict[self.agents[i_agent].kind]
            self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = 0.0
            self.view[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]][0] = self.agents[i_agent].color

            if self.agents[i_agent].group == 0 :
               self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = 0.0
               self.view_3ch[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]][0] = self.agents[i_agent].color
            elif self.agents[i_agent].group == 1 :
               self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][2] = 0.0
               self.view_3ch[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]][2] = self.agents[i_agent].color
            self.pos_to_object[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = None
            self.pos_to_object[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] = self.agents[i_agent]
            self.agents[i_agent].reward = self.agents[i_agent].reward_candidate
            self.agents[i_agent].pos = self.agents[i_agent].new_pos
         else:
            self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            if self.agents[i_agent].group == 0 :
               self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            elif self.agents[i_agent].group == 1 :
               self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][2] = self.agents[i_agent].color
            self.agents[i_agent].reward = 0.0

         self.agents[i_agent].total_reward += self.agents[i_agent].reward
         self.agents[i_agent].move_permission = 'no'

         #update of memory matrices
         if t >= 1 :
            Xi_temp = self.agents[i_agent].Xi_temp_part0 - self.agents[i_agent].gamma*self.agents[i_agent].Xi_temp_part1
            if self.experience_sharing == 'on':
               self.agents[0].reward_list.append(self.agents[i_agent].reward_past)
               self.agents[0].Xi_temp_list.append(Xi_temp)
               self.agents[0].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0)
            elif self.experience_sharing == 'off':
               self.agents[i_agent].reward_list.append(self.agents[i_agent].reward_past)
               self.agents[i_agent].Xi_temp_list.append(Xi_temp)
               self.agents[i_agent].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0)

         if self.done:
            #calculation on the last step
            delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
            dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
            state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]

            action_past = action
            self.agents[i_agent].Xi_temp_part0 = copy.deepcopy(self.agents[i_agent].Xi_temp_part1)
            self.agents[i_agent].reward_past = self.agents[i_agent].reward
            self.agents[i_agent].action, self.agents[i_agent].Xi_temp_part1, self.agents[i_agent].Q_temp = self.agents[i_agent].choose_action(state)
            
            Xi_temp = self.agents[i_agent].Xi_temp_part0 - self.agents[i_agent].gamma*self.agents[i_agent].Xi_temp_part1

            if self.experience_sharing == 'on':
               self.agents[0].reward_list.append(self.agents[i_agent].reward_past)
               self.agents[0].Xi_temp_list.append(Xi_temp)
               self.agents[0].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0) 
            elif self.experience_sharing == 'off':
               self.agents[i_agent].reward_list.append(self.agents[i_agent].reward_past)
               self.agents[i_agent].Xi_temp_list.append(Xi_temp)
               self.agents[i_agent].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part0) 

            if self.experience_sharing == 'on':
               self.agents[0].reward_list.append(0.0)
               self.agents[0].Xi_temp_list.append(self.agents[i_agent].Xi_temp_part1)
               self.agents[0].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part1)
            elif self.experience_sharing == 'off':
               self.agents[i_agent].reward_list.append(0.0)
               self.agents[i_agent].Xi_temp_list.append(self.agents[i_agent].Xi_temp_part1)
               self.agents[i_agent].Xi_temp_p0_list.append(self.agents[i_agent].Xi_temp_part1)


      if self.done and self.experience_sharing == 'on':
         self.agents[0].calculate_Wout()
         for i_agent in range(self.n_agent):
            self.agents[i_agent].W_out = copy.deepcopy(self.agents[0].W_out)
      if self.done and self.experience_sharing == 'off':
         for i_agent in range(self.n_agent):
            self.agents[i_agent].calculate_Wout()

      if self.done:
         for i_agent in range(self.n_agent):
            if self.agents[i_agent].epsilon > self.epsilon_min:
               self.agents[i_agent].epsilon *= self.agents[i_agent].epsilon_decay

      #resetting move_candidate_number
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)

   def _render(self):
      if self.render_mode == 'CUI' :
         for j in range(self.height):
            for i in range(self.width):
               if self.condition[i][j] == condition_dict["agent"] :
                  print("A", end="")
               elif self.condition[i][j] == condition_dict["wall"] :
                  print("X", end="")
               else:
                  print(" ", end="")
            print()
         print(self.agents[0].reward, self.agents[1].reward)
      elif self.render_mode == 'rgb_array' :
         view_3ch_temp = self.view_3ch.transpose(1,0,2)
         view_3ch_temp = np.maximum( 255.0*np.minimum( view_3ch_temp, np.ones(view_3ch_temp.shape) ), np.zeros(view_3ch_temp.shape) ).astype(np.uint8)
         self.Image_list.append( Image.fromarray( np.repeat(np.repeat(view_3ch_temp,5,axis=0),5,axis=1) ) )
      elif self.render_mode == 'off' :
         pass

   def _rendermode_change_to_rgb(self):
      self.render_mode = 'rgb_array'
      self.Image_list = []

class Object:
   def __init__(self,x,y,condition, kind, hyperparameters):
      self.hyperparameters = hyperparameters
      self.action_space = [0,0] 
      self.v = 1
      self.vskew = 1
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.kind = kind
      self.define_orientation = {"left":0, "right":1}
      self.orientation = 0
      self.attack_switch = 0     
      self.attack_existence = False
      self.agent_eyesight_whole = hyperparameters.agent_eyesight_whole 
      self.agent_eyesight_1side = self.agent_eyesight_whole//2

      self.attacked_count = 0
      self.attacker = []
      self.move_permission = 'no'

      self.pos = np.array([x,y])
      self.color = 1.0
      condition[self.pos[0]][self.pos[1]] = condition_dict[kind]

   def _key_step(self,pos_to_object):
      print("Choose 'u','d','r', 'l', 'au', 'ad', 'ar', or 'al':")
      key_command = input()
      if key_command == 'u':
         action = 0
      elif key_command == 'd':
         action = 1
      elif key_command == 'r':
         action = 2
      elif key_command == 'l':
         action = 3
      else:
         action = 0
      self._step(action,pos_to_object)

   def _step(self, action, pos_to_object):
      #action[0]...direction of the movement, action[1]:attacking
      if action == 0:
         direction = np.array([1,0])
      elif action == 1:
         direction = np.array([0,1])
      elif action == 2:
         direction = np.array([-1,0])
      elif action == 3:
         direction = np.array([0,-1])
      self.new_pos = self.pos + direction
      if self.__class__.__name__ == 'Agent' :
         if self.group == 0 : 
            self.reward_candidate = float(direction[0])
         elif self.group == 1 : 
            self.reward_candidate = -float(direction[0])
      #Periodic boundary condition
      if self.new_pos[0] < 0:
         self.new_pos[0] += self.width
      elif self.new_pos[0] >= self.width:
         self.new_pos[0] -= self.width
      if self.new_pos[1] < 0:
         self.new_pos[1] += self.height
      elif self.new_pos[1] >= self.height:
         self.new_pos[1] -= self.height
      #Update of the position itself is executed by Env.step().

class Agent(Object):
   def __init__(self,x,y,condition, kind, hyperparameters):
      super().__init__(x,y,condition, kind, hyperparameters)
      # building neural net
      self.state_size = self.agent_eyesight_whole*self.agent_eyesight_whole*2
      self.action_size = 4
      self.action_choice = [_ for _ in range (self.action_size)] 
      self.sparsity = self.hyperparameters.sparsity
      self.central_sparsity = self.hyperparameters.central_sparsity
      self.core_sparsity = self.hyperparameters.core_sparsity
      self.spectral_radius = self.hyperparameters.spectral_radius
      self.lr = self.hyperparameters.leaking_rate
      self.gamma = self.hyperparameters.gamma
      self.beta = self.hyperparameters.beta # Ridge term 
      self.reservoir_size = self.hyperparameters.reservoir_size
      self.reservoir_output_size = self.reservoir_size + 1
      self.epsilon = self.hyperparameters.epsilon #epsilon-greedy
      self.epsilon_decay = self.hyperparameters.epsilon_decay
      self.decay_ratio = self.hyperparameters.decay_ratio #decay of memory matrices per one update of W_out

      self.n_group = self.hyperparameters.n_group

      self.reward_list = []
      self.Xi_temp_list = []
      self.Xi_temp_p0_list = []

      self.total_reward = 0.0
      self.reward = 0.0
      self.group = 0
      self.sigma_W_in_a = self.hyperparameters.sigma_W_in_a

      self.W_in_s = np.random.normal(loc=0.0,scale=1.0,size=(self.reservoir_size,self.state_size+1))
      self.W_in_a = np.random.normal(loc=0.0,scale=self.sigma_W_in_a,size=(self.reservoir_size,self.action_size))
      self.W_in_group = np.random.normal(loc=0.0,scale=self.sigma_W_in_a,size=(self.reservoir_size,self.n_group))
      self.W_res = np.random.normal(loc=0.0,scale=1.0,size=(self.reservoir_size,self.reservoir_size))
      self.W_out = np.random.normal(loc=0.0,scale=0.0,size=(1,self.reservoir_output_size) )
      self.X_esn = np.zeros(self.reservoir_size)

      Cut_prob_W_in_s = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.state_size+1))
      Cut_prob_W_in_a = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.action_size))
      Cut_prob_W_res = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.reservoir_size))

      Cut_prob_mat = self.sparsity * np.ones( (self.agent_eyesight_whole,self.agent_eyesight_whole,2) )
      Cut_prob_mat[self.agent_eyesight_1side-3:self.agent_eyesight_1side+4,self.agent_eyesight_1side-3:self.agent_eyesight_1side+4,:] = self.central_sparsity
      Cut_prob_mat[self.agent_eyesight_1side-1:self.agent_eyesight_1side+2,self.agent_eyesight_1side-1:self.agent_eyesight_1side+2,:] = self.core_sparsity
      Cut_prob_mat = np.concatenate( [Cut_prob_mat.reshape(-1), [self.sparsity]] )

      for i_1 in range(self.W_in_s.shape[0]):
         for i_2 in range(self.W_in_s.shape[1]):
            if Cut_prob_W_in_s[i_1][i_2] < Cut_prob_mat[i_2] :
               self.W_in_s[i_1][i_2] = 0.00
      for i_1 in range(self.W_res.shape[0]):
         for i_2 in range(self.W_res.shape[1]):
            if Cut_prob_W_res[i_1][i_2] < self.sparsity:
               self.W_res[i_1][i_2] = 0.00

      specrad_temp_W_res = np.max( abs(np.linalg.eigvals(self.W_res)) )
      self.W_res *= self.spectral_radius/specrad_temp_W_res

      self.W_in_s = csr_matrix(self.W_in_s)
      self.W_res = csr_matrix(self.W_res)

      self.XXT = self.beta*np.identity(self.reservoir_output_size)
      self.rX = np.zeros((1, self.reservoir_output_size))


   #methods related with reinforcement learning
   def _ReLU(self,x):
      zeros = np.zeros(x.shape)
      return np.maximum(x,zeros)

   def _softmax(self,z):
      z_rescale = self.inverse_temperature*(z - np.max(z))
      z_exp = np.exp(np.clip(z_rescale,-250,250))
      return z_exp/( np.sum(z_exp) )

   def choose_action(self, state): #epsilon-greedy
      self.X_in = np.concatenate( [state.reshape(-1), [1.0]] )
      Input_state = np.repeat( ((self.W_in_s*self.X_in) + (self.W_res*self.X_esn) + self.W_in_group[:,self.group]).reshape(-1,1), self.action_size, axis=1)
      self.X_esn_tilde = self._ReLU( Input_state + self.W_in_a )
      X_esn_candidates = np.repeat(self.X_esn.reshape(-1,1), self.action_size, axis=1) + self.lr*(self.X_esn_tilde - np.repeat(self.X_esn.reshape(-1,1), self.action_size, axis=1) )
      self.Q = np.dot(self.W_out,np.concatenate([X_esn_candidates, np.ones((1,self.action_size))],axis=0) ).reshape(-1)

      p = np.random.uniform(low=0.0,high=1.0)
      if p < self.epsilon:
         action_chosen = np.random.choice(self.action_choice) 
      else:
         action_chosen = np.argmax(self.Q) 
      self.X_esn = X_esn_candidates[:,action_chosen]
      return action_chosen, copy.deepcopy( np.concatenate([self.X_esn,[1.0]]) ), self.Q[action_chosen]  # returns action and corresponding X_res

   def calculate_Wout(self): 
      self.rX += np.dot(np.array(self.reward_list).reshape(1,-1), np.array(self.Xi_temp_p0_list) )
      self.XXT += np.dot(np.array(self.Xi_temp_list).T , np.array(self.Xi_temp_p0_list) )
      XXTinv = np.linalg.inv( self.XXT )
      self.W_out = np.dot(self.rX,XXTinv)
      self.rX *= self.decay_ratio
      self.XXT *= self.decay_ratio
      self.reward_list = []
      self.Xi_temp_list = []
      self.Xi_temp_p0_list = []

   def _onehot(self, action):
      onehot : array # shape = (1,self.action_size)
      onehot = np.zeros( (self.action_size,1) )
      onehot[action,0] = 1.0
      return onehot

   def action_reservoir_matrix(self, action, x_esn):
      delta = self._onehot(action)
      xi = np.dot(delta, x_esn.reshape(1,-1))
      return xi.reshape(1,-1)



if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.txt"
   filename_density = "density_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.txt"
   file_time = open(filename_time, "w")

   t1 = 0
   myEnv = Env(hyperparameters=hyper)
   total_reward_list = []
   mean_view = np.zeros((myEnv.width,myEnv.height,3))
   var_view = np.zeros((myEnv.width,myEnv.height,3))
   mean_count = 0.0
   #print( myEnv.condition[myEnv.agents[0].pos[0]][myEnv.agents[0].pos[1]] )
   for n_episode in range (ep_termination):
      myEnv.done = False
      for t1 in range (t_max):
         myEnv._render()
         if t1 == t_max-1:
            myEnv.done = True
         if n_episode >= ep_observe and t1 >= t_observe:
            mean_view += myEnv.view_3ch
            var_view += myEnv.view_3ch*myEnv.view_3ch
            mean_count += 1.0
         myEnv._step(t1)

      reward_list = np.zeros(myEnv.n_agent)
      for i1 in range (myEnv.n_agent):         
         print(myEnv.agents[i1].total_reward )
         reward_list[i1] = myEnv.agents[i1].total_reward
      print(np.mean(reward_list), np.max(reward_list), np.min(reward_list) )
      print(n_episode+1, np.mean(reward_list), np.max(reward_list), np.min(reward_list), sep="	", file=file_learning_curve)
      print(n_episode+1, 'th learning ...')
      if n_episode >= ep_observe:
         total_reward_list.append(myEnv.agents[0].total_reward)
      myEnv.reset()

   mean_view /= mean_count
   var_view -= mean_count*mean_view*mean_view
   err_view = np.sqrt(var_view/(mean_count*(mean_count-1.0)) )
   for j in range(myEnv.height):
      for i in range(myEnv.width):
         print(i,myEnv.height - 1 - j, mean_view[i,j,0], err_view[i,j,0], mean_view[i,j,1], err_view[i,j,1], mean_view[i,j,2], err_view[i,j,2], sep="	", file=file_density )
      print("", file=file_density)

   file_learning_curve.close()
   file_density.close()
   
   #gif-animation
   filename_gif = "animation_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.gif"
   
   filename_snap1 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_0_" + str(seed) + "th_try.png"
   filename_snap2 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_100_" + str(seed) + "th_try.png"
   filename_snap3 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_" + str(t_max-1) + "_" + str(seed) + "th_try.png"
   myEnv._rendermode_change_to_rgb()
   myEnv.reset()
   for t1 in range (t_max):
      myEnv._render()
      myEnv._step(t1)
      #time.sleep(0.5)
   if myEnv.render_mode == 'rgb_array':
      myEnv.Image_list[0].save(filename_gif, save_all=True, append_images=myEnv.Image_list[1:], optimize=False, duration=200, loop=0)
      myEnv.Image_list[0].save(filename_snap1)
      myEnv.Image_list[100].save(filename_snap2)
      myEnv.Image_list[t_max-1].save(filename_snap3)
   print(myEnv.agents[0].total_reward)
   myEnv.agents[0].total_reward = 0.0

   
   #density plot
   filename_colormap = "colormap_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "group1_" + str(seed) + "th_try.png"
   mean_view_white= (255.0*mean_view.transpose(1,0,2)).astype(np.uint8)
   mean_view_white[:,:,2] = np.where(mean_view_white[:,:,1] == 0, 255 - mean_view_white[:,:,0], 0)
   mean_view_white[:,:,1] = np.maximum( 255 - mean_view_white[:,:,0] , mean_view_white[:,:,1] )
   mean_view_white[:,:,0] = 255*np.where(mean_view_white[:,:,1] == mean_view_white[:,:,2], 1, 0)
   mean_Image = Image.fromarray( np.repeat(np.repeat(mean_view_white,5,axis=0),5,axis=1) )
   mean_Image.save(filename_colormap)

   filename_colormap = "colormap_" + str(hyper.n_agent) + "pedestrians_task2_1group_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "group2_" + str(seed) + "th_try.png"
   mean_view_white = (255.0*mean_view.transpose(1,0,2)).astype(np.uint8)
   mean_view_white[:,:,0] = np.where(mean_view_white[:,:,1] == 0, 255 - mean_view_white[:,:,2], 0)
   mean_view_white[:,:,1] = np.maximum( 255 - mean_view_white[:,:,2] , mean_view_white[:,:,1] )
   mean_view_white[:,:,2] = 255*np.where(mean_view_white[:,:,1] == mean_view_white[:,:,0], 1, 0)
   mean_Image = Image.fromarray( np.repeat(np.repeat(mean_view_white,5,axis=0),5,axis=1) )
   mean_Image.save(filename_colormap)

   print( np.mean( np.array(total_reward_list) ) )
   for i1 in total_reward_list:
      print(i1)

   print(time.time() - start_time, file=file_time)
   