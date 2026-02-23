"""
Source code for task.II
2025/9/29 improved the forward propagation to save the computational time
2025/10/3 changed the dtype of NN parameters into np.float32
2026/1/12 improved efficiency of the forward propagation
*activation function can be chosen as a hyperparameter
"""
import numpy as np
import time
import copy
from PIL import Image
from scipy.sparse import csc_matrix

start_time = time.time()
#np.set_printoptions(precision=3) # used for debugging

seed = 1
np.random.seed(seed)

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

      self.activation = 'relu' # 'relu' or 'tanh'


class Env:
   def __init__(self, hyperparameters, render_mode='off'):
      #render_mode : 'off' or 'rgb_array'
      self.render_mode = render_mode
      # This code assumes experience_sharing.
      if self.render_mode == 'rgb_array' :
         self.Image_list = []

      self.agents = []
      self.walls = []
      self.hyperparameters = hyperparameters
      self.road_width = self.hyperparameters.road_width
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.view_3ch = np.zeros((self.width,self.height,3))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
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
         elif i_agent in {1, self.n_agent_1group}:
            new_agent = copy.deepcopy(self.agents[0]) # agents[1] and agents[self.n_agent_1group] copies agents[0] without/with neunet.
            new_agent.pos = pos_temp_agent
            if i_agent == 1:
               self.agents[0].make_neunet()
         else:
            new_agent = copy.deepcopy(self.agents[1])
            new_agent.pos = pos_temp_agent
         if i_agent <= self.n_agent_1group - 1 :
            new_agent.group = 0
            self.view_3ch[new_agent.pos[0]][new_agent.pos[1]][0] = new_agent.color
         else:
            new_agent.group = 1
            self.view_3ch[new_agent.pos[0]][new_agent.pos[1]][2] = new_agent.color
         self.agents.append(new_agent)
         self.condition[new_agent.pos[0]][new_agent.pos[1]] = condition_dict["agent"]
         self.view[new_agent.pos[0]][new_agent.pos[1]][0] = self.agents[i_agent].color

      self.agents[0].X_esn = np.zeros((self.n_agent_1group, self.agents[0].reservoir_size), dtype=np.float32)
      self.agents[self.n_agent_1group].X_esn = np.zeros((self.n_agent_1group, self.agents[self.n_agent_1group].reservoir_size), dtype=np.float32)

      self.whole_action = [None, None]
      self.whole_Xi_temp_part1 = [None, None]
      self.whole_Xi_temp = [None, None]

   def agent_position(self, i_agent):
      if i_agent <= self.n_agent_1group - 1 :
         return np.array([0+(i_agent%2)+2*(i_agent//self.road_width),7+(i_agent%self.road_width)])
      elif self.n_agent_1group <= i_agent <= self.n_agent - 1 :
         return np.array([19-(i_agent%2)-2*((i_agent-self.n_agent_1group)//self.road_width),7+((i_agent-self.n_agent_1group)%self.road_width)])

   def build_wall(self):
      if self.walls: 
         for w1 in self.walls:
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
      new_wall = Object(x,y, self.condition, kind=("wall"))
      self.walls.append(new_wall)
      self.view[x][y][1] = new_wall.color
      self.view_3ch[x][y][1] = new_wall.color

   def reset(self):
      #reset of environment with keeping the result of training
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.view_3ch = np.zeros((self.width,self.height,3))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.done = False
      self.build_wall()
      self.agents[0].X_esn = np.zeros((self.n_agent_1group, self.agents[0].reservoir_size), dtype=np.float32)
      self.agents[self.n_agent_1group].X_esn = np.zeros((self.n_agent_1group, self.agents[self.n_agent_1group].reservoir_size), dtype=np.float32)
      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         self.agents[i_agent].pos = pos_temp_agent
         self.agents[i_agent].total_reward = 0.0
         self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = condition_dict["agent"]
         self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
         if self.agents[i_agent].group == 0 :
            self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
         elif self.agents[i_agent].group == 1 :
            self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][2] = self.agents[i_agent].color

   def _step(self, t):
      if t >= 1:
         whole_Xi_temp_part0 = self.whole_Xi_temp_part1.copy() 
         whole_reward_past = self.whole_reward.copy()

      #agents' action
      # getting batched observation 
      x_id_batch = [None, None]
      y_id_batch = [None, None]
      state_array = [None, None]
      whole_pos_x = np.array([agent.pos[0] for agent in self.agents]) 
      whole_pos_y = np.array([agent.pos[1] for agent in self.agents]) 
      x_id_batch[0] = (whole_pos_x[:self.n_agent_1group, None] + self.agents[0].offset[None, :]) % self.width
      y_id_batch[0] = (whole_pos_y[:self.n_agent_1group, None] + self.agents[0].offset[None, :]) % self.height
      x_id_batch[1] = (whole_pos_x[self.n_agent_1group:, None] + self.agents[self.n_agent_1group].offset[None, :]) % self.width
      y_id_batch[1] = (whole_pos_y[self.n_agent_1group:, None] + self.agents[self.n_agent_1group].offset[None, :]) % self.height
      state_array[0] = self.view[x_id_batch[0][:, :, None], y_id_batch[0][:, None, :]].astype(np.float32).reshape(-1, self.agents[0].state_size)
      state_array[1] = self.view[x_id_batch[1][:, :, None], y_id_batch[1][:, None, :]].astype(np.float32).reshape(-1, self.agents[self.n_agent_1group].state_size)

      self.whole_action[0], self.whole_Xi_temp_part1[0] = self.agents[0].choose_action(state_array[0])
      self.whole_action[1], self.whole_Xi_temp_part1[1] = self.agents[self.n_agent_1group].choose_action(state_array[1])
      for i_agent in range(self.n_agent):
         self.agents[i_agent].action = self.whole_action[self.agents[i_agent].group][i_agent % self.n_agent_1group]
         self.agents[i_agent]._step(self.agents[i_agent].action)
         self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] += 1   

      #update of the positions of agents
      for i_agent in range(self.n_agent):
         if self.condition[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == condition_dict["vacant"] and self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] == 1 :
            self.agents[i_agent].move_permission = 'yes'

      self.whole_reward = [[],[]]
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
            self.agents[i_agent].reward = self.agents[i_agent].reward_candidate
            self.agents[i_agent].pos = self.agents[i_agent].new_pos
         else:
            self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            if self.agents[i_agent].group == 0 :
               self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            elif self.agents[i_agent].group == 1 :
               self.view_3ch[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][2] = self.agents[i_agent].color
            self.agents[i_agent].reward = 0.0

         self.whole_reward[self.agents[i_agent].group].append(self.agents[i_agent].reward)
         self.agents[i_agent].total_reward += self.agents[i_agent].reward
         self.agents[i_agent].move_permission = 'no'

      #update of memory matrices
      if t >= 1 :
         self.whole_Xi_temp[0] = whole_Xi_temp_part0[0] - self.agents[0].gamma*self.whole_Xi_temp_part1[0]
         self.whole_Xi_temp[1] = whole_Xi_temp_part0[1] - self.agents[self.n_agent_1group].gamma*self.whole_Xi_temp_part1[1]

         self.agents[0].reward_list.append(whole_reward_past[0])
         self.agents[0].Xi_temp_list.append(self.whole_Xi_temp[0])
         self.agents[0].Xi_temp_p0_list.append(whole_Xi_temp_part0[0]) 
         self.agents[self.n_agent_1group].reward_list.append(whole_reward_past[1])
         self.agents[self.n_agent_1group].Xi_temp_list.append(self.whole_Xi_temp[1])
         self.agents[self.n_agent_1group].Xi_temp_p0_list.append(whole_Xi_temp_part0[1]) 

      if self.done:
         #calculation on the last step
         # getting batched observation 
         x_id_batch = [None, None]
         y_id_batch = [None, None]
         state_array = [None, None]
         whole_pos_x = np.array([agent.pos[0] for agent in self.agents]) 
         whole_pos_y = np.array([agent.pos[1] for agent in self.agents]) 
         x_id_batch[0] = (whole_pos_x[:self.n_agent_1group, None] + self.agents[0].offset[None, :]) % self.width
         y_id_batch[0] = (whole_pos_y[:self.n_agent_1group, None] + self.agents[0].offset[None, :]) % self.height
         x_id_batch[1] = (whole_pos_x[self.n_agent_1group:, None] + self.agents[self.n_agent_1group].offset[None, :]) % self.width
         y_id_batch[1] = (whole_pos_y[self.n_agent_1group:, None] + self.agents[self.n_agent_1group].offset[None, :]) % self.height
         state_array[0] = self.view[x_id_batch[0][:, :, None], y_id_batch[0][:, None, :]].astype(np.float32).reshape(-1, self.agents[0].state_size)
         state_array[1] = self.view[x_id_batch[1][:, :, None], y_id_batch[1][:, None, :]].astype(np.float32).reshape(-1, self.agents[self.n_agent_1group].state_size)

         whole_reward_past = self.whole_reward.copy()
         whole_Xi_temp_part0 = self.whole_Xi_temp_part1.copy()

         self.whole_action[0], self.whole_Xi_temp_part1[0] = self.agents[0].choose_action(state_array[0])
         self.whole_Xi_temp[0] = whole_Xi_temp_part0[0] - self.agents[0].gamma*self.whole_Xi_temp_part1[0]
         self.whole_action[1], self.whole_Xi_temp_part1[1] = self.agents[self.n_agent_1group].choose_action(state_array[1])
         self.whole_Xi_temp[1] = whole_Xi_temp_part0[1] - self.agents[self.n_agent_1group].gamma*self.whole_Xi_temp_part1[1]

         self.agents[0].reward_list.append(whole_reward_past[0])
         self.agents[0].Xi_temp_list.append(self.whole_Xi_temp[0])
         self.agents[0].Xi_temp_p0_list.append(whole_Xi_temp_part0[0]) 
         self.agents[self.n_agent_1group].reward_list.append(whole_reward_past[1])
         self.agents[self.n_agent_1group].Xi_temp_list.append(self.whole_Xi_temp[1])
         self.agents[self.n_agent_1group].Xi_temp_p0_list.append(whole_Xi_temp_part0[1]) 

         self.agents[0].reward_list.append([0.0 for _ in range(self.n_agent_1group)])
         self.agents[0].Xi_temp_list.append(self.whole_Xi_temp_part1[0])
         self.agents[0].Xi_temp_p0_list.append(self.whole_Xi_temp_part1[0])
         self.agents[self.n_agent_1group].reward_list.append([0.0 for _ in range(self.n_agent_1group)])
         self.agents[self.n_agent_1group].Xi_temp_list.append(self.whole_Xi_temp_part1[1])
         self.agents[self.n_agent_1group].Xi_temp_p0_list.append(self.whole_Xi_temp_part1[1])

         self.agents[0].calculate_Wout()
         self.agents[self.n_agent_1group].calculate_Wout()

         if self.agents[0].epsilon > self.epsilon_min:
            self.agents[0].epsilon *= self.agents[0].epsilon_decay
         if self.agents[self.n_agent_1group].epsilon > self.epsilon_min:
            self.agents[self.n_agent_1group].epsilon *= self.agents[self.n_agent_1group].epsilon_decay

      #resetting move_candidate_number
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)

   def _render(self):
      if self.render_mode == 'rgb_array' :
         view_3ch_temp = self.view_3ch.transpose(1,0,2)
         view_3ch_temp = np.maximum( 255.0*np.minimum( view_3ch_temp, np.ones(view_3ch_temp.shape) ), np.zeros(view_3ch_temp.shape) ).astype(np.uint8)
         self.Image_list.append( Image.fromarray( np.repeat(np.repeat(view_3ch_temp,5,axis=0),5,axis=1) ) )

   def _rendermode_change_to_rgb(self):
      self.render_mode = 'rgb_array'
      self.Image_list = []

class Object:
   def __init__(self,x,y,condition, kind):
      self.kind = kind
      self.pos = np.array([x,y])
      self.color = 1.0
      condition[self.pos[0]][self.pos[1]] = condition_dict[kind]


class Agent(Object):
   def __init__(self,x,y,condition, kind, hyperparameters):
      super().__init__(x,y,condition, kind)
      self.hyperparameters = hyperparameters
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.move_permission = 'no'
      self.total_reward = 0.0

   def make_neunet(self): # lazy initialization for particular agents' attribute
      self.agent_eyesight_whole = self.hyperparameters.agent_eyesight_whole 
      self.agent_eyesight_1side = self.agent_eyesight_whole//2
      self.state_size = self.agent_eyesight_whole*self.agent_eyesight_whole*2
      self.offset = np.arange(-self.agent_eyesight_1side, self.agent_eyesight_1side + 1)
      self.action_size = 4
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

      self.reward_list = []
      self.Xi_temp_list = []
      self.Xi_temp_p0_list = []

      self.sigma_W_in_a = self.hyperparameters.sigma_W_in_a

      self.activation = self.hyperparameters.activation

      self.W_in_s = np.random.normal(loc=0.0,scale=1.0,size=(self.reservoir_size,self.state_size+1)).astype(np.float32)
      self.W_in_a = np.random.normal(loc=0.0,scale=self.sigma_W_in_a,size=(self.reservoir_size,self.action_size)).astype(np.float32)
      self.W_res = np.random.normal(loc=0.0,scale=1.0,size=(self.reservoir_size,self.reservoir_size)).astype(np.float32)
      self.W_out = np.zeros((self.reservoir_output_size,), dtype=np.float32)

      Cut_prob_W_in_s = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.state_size+1)).astype(np.float32)
      Cut_prob_W_res = np.random.uniform(low=0.0,high=1.0,size=(self.reservoir_size,self.reservoir_size)).astype(np.float32)

      Cut_prob_mat = self.sparsity * np.ones( (self.agent_eyesight_whole,self.agent_eyesight_whole,2), dtype=np.float32 )
      Cut_prob_mat[self.agent_eyesight_1side-3:self.agent_eyesight_1side+4,self.agent_eyesight_1side-3:self.agent_eyesight_1side+4,:] = self.central_sparsity
      Cut_prob_mat[self.agent_eyesight_1side-1:self.agent_eyesight_1side+2,self.agent_eyesight_1side-1:self.agent_eyesight_1side+2,:] = self.core_sparsity
      Cut_prob_mat = np.concatenate( [Cut_prob_mat.reshape(-1), [self.sparsity]], dtype=np.float32 )

      self.W_in_s = np.where(Cut_prob_W_in_s < Cut_prob_mat, 0.00, self.W_in_s)
      self.W_res = np.where(Cut_prob_W_res < self.sparsity, 0.00, self.W_res)

      specrad_temp_W_res = np.max( abs(np.linalg.eigvals(self.W_res)) )
      self.W_res *= self.spectral_radius/specrad_temp_W_res

      self.W_in_s = csc_matrix(self.W_in_s)
      self.W_res = csc_matrix(self.W_res)

      self.XXT = self.beta*np.identity(self.reservoir_output_size) # Matrices related with calulation of inv_mat should not be float32.
      self.rX = np.zeros((self.reservoir_output_size,))


   def _step(self, action): # moved from Object._step() (refactor only; behaviors unchanged)
      if action == 0:
         direction = np.array([1,0])
      elif action == 1:
         direction = np.array([0,1])
      elif action == 2:
         direction = np.array([-1,0])
      elif action == 3:
         direction = np.array([0,-1])
      self.new_pos = self.pos + direction
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

   #methods related with reinforcement learning
   def _ReLU(self,x):
      return np.maximum(x, 0.0)

   def choose_action(self, state_arr): #epsilon-greedy
      self.X_in = np.concatenate( [state_arr, np.ones((state_arr.shape[0],1), dtype=np.float32)], axis=1 )
      Input_state = np.repeat( (self.X_in * self.W_in_s.T + self.X_esn * self.W_res.T)[:,np.newaxis,:], self.action_size, axis=1) # T*a*Nres array
      if self.activation == 'relu':
         self.X_esn_tilde = self._ReLU( Input_state + self.W_in_a.T[np.newaxis,:,:] )
      elif self.activation == 'tanh': 
         self.X_esn_tilde = np.tanh( Input_state + self.W_in_a.T[np.newaxis,:,:] )
      X_esn_past = self.X_esn[:,np.newaxis,:] 
      X_esn_candidates = X_esn_past + self.lr*(self.X_esn_tilde - X_esn_past) # Nres*T*a array
      self.Q = np.dot( np.concatenate([X_esn_candidates, np.ones((X_esn_candidates.shape[0],self.action_size,1), dtype=np.float32)],axis=2), self.W_out ) #T*a array

      p = np.random.uniform(low=0.0,high=1.0, size = (self.Q.shape[0],)) 
      action_argmax = np.argmax(self.Q, axis=1) 
      action_random = np.random.randint(self.action_size, high=None, size=(self.Q.shape[0],))
      action_chosen = np.where(p < self.epsilon, action_random, action_argmax)
      self.X_esn = X_esn_candidates[np.arange(X_esn_candidates.shape[0]),action_chosen,:]
      return action_chosen, np.concatenate([self.X_esn, np.ones((self.X_esn.shape[0],1), dtype=np.float32)],axis=1) # returns action and corresponding X_res

   def calculate_Wout(self): 
      reward_array_1 = np.concatenate(self.reward_list, axis=0, dtype=np.float32)
      Xi_temp_array_1 = np.concatenate(self.Xi_temp_list, axis=0)
      Xi_temp_p0_array_1 = np.concatenate(self.Xi_temp_p0_list, axis=0)
      self.rX += np.dot(reward_array_1, Xi_temp_p0_array_1 )
      self.XXT += np.dot(Xi_temp_array_1.T , Xi_temp_p0_array_1 )
      XXTinv = np.linalg.inv( self.XXT )
      self.W_out = np.dot(self.rX,XXTinv).astype(np.float32)
      self.rX *= self.decay_ratio
      self.XXT *= self.decay_ratio
      self.reward_list = []
      self.Xi_temp_list = []
      self.Xi_temp_p0_list = []



if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.txt"
   filename_density = "density_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.txt"
   file_time = open(filename_time, "w")

   t1 = 0
   myEnv = Env(hyperparameters=hyper)
   total_reward_list = []
   mean_view = np.zeros((myEnv.width,myEnv.height,3))
   var_view = np.zeros((myEnv.width,myEnv.height,3))
   mean_count = 0.0
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
   filename_gif = "animation_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "th_try.gif"
   
   filename_snap1 = "snapshot_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_0_" + str(seed) + "th_try.png"
   filename_snap2 = "snapshot_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_100_" + str(seed) + "th_try.png"
   filename_snap3 = "snapshot_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_" + str(t_max-1) + "_" + str(seed) + "th_try.png"
   myEnv._rendermode_change_to_rgb()
   myEnv.reset()
   for t1 in range (t_max):
      myEnv._render()
      myEnv._step(t1)
   if myEnv.render_mode == 'rgb_array':
      myEnv.Image_list[0].save(filename_gif, save_all=True, append_images=myEnv.Image_list[1:], optimize=False, duration=200, loop=0)
      myEnv.Image_list[0].save(filename_snap1)
      myEnv.Image_list[100].save(filename_snap2)
      myEnv.Image_list[t_max-1].save(filename_snap3)
   print(myEnv.agents[0].total_reward)
   myEnv.agents[0].total_reward = 0.0
   
   #density plot
   filename_colormap = "colormap_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "group1_" + str(seed) + "th_try.png"
   mean_view_white= (255.0*mean_view.transpose(1,0,2)).astype(np.uint8)
   mean_view_white[:,:,2] = np.where(mean_view_white[:,:,1] == 0, 255 - mean_view_white[:,:,0], 0)
   mean_view_white[:,:,1] = np.maximum( 255 - mean_view_white[:,:,0] , mean_view_white[:,:,1] )
   mean_view_white[:,:,0] = 255*np.where(mean_view_white[:,:,1] == mean_view_white[:,:,2], 1, 0)
   mean_Image = Image.fromarray( np.repeat(np.repeat(mean_view_white,5,axis=0),5,axis=1) )
   mean_Image.save(filename_colormap)

   filename_colormap = "colormap_" + hyper.activation + "_batch_forward_f32_" + str(hyper.n_agent) + "pedestrians_task2_modified_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.reservoir_size) + "_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "group2_" + str(seed) + "th_try.png"
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
   