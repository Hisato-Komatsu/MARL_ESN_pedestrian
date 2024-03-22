"""
Source code for task.II using DQN
"""
import random
import numpy as np
import time
import os
import copy
from PIL import Image, ImageDraw
from collections import deque

import torch
from torch import nn
import torch.nn.functional as torF
from torch.distributions import Categorical

start_time = time.time()
np.set_printoptions(precision=3)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

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
      self.n_agent = 32 #the number of agents
      self.agent_eyesight_whole = 11 #odd number

      self.gamma = 0.95
      self.layer_size = 1024
      self.epsilon = 1.0 #epsilon-greedy
      self.epsilon_decay = 0.95
      self.epsilon_min = 0.02

      self.n_group = 2 #number of groups of agents

      self.learning_rate = 0.00025
      self.batch_size = 64
      self.memory_size = 1000000
      self.max_grad_norm = 50.0


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
      
      self.gamma = self.hyperparameters.gamma
      self.learning_rate = self.hyperparameters.learning_rate
      self.max_grad_norm = self.hyperparameters.max_grad_norm

      self.build_wall()

      self.net = [_ for _ in range(self.n_group)]
      self.optimizer = [_ for _ in range(self.n_group)]
      for i_group in range(self.n_group):
         self.net[i_group] = WholeNet(hyperparameters=self.hyperparameters)
         self.optimizer[i_group] = torch.optim.Adam(self.net[i_group].main_net.parameters(), lr=self.learning_rate)

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
         if t == 0:
            delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
            dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
            state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
            self.agents[i_agent].state = torch.from_numpy(state)
         else:
            self.agents[i_agent].state = self.agents[i_agent].next_state
         with torch.no_grad() :
            q_temp = self.net[self.agents[i_agent].group].forward( self.agents[i_agent].state[None, :] )
         self.agents[i_agent].action = self.net[self.agents[i_agent].group].choose_action(q_temp)
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

      for i_agent in range(self.n_agent):
         delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
         dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
         next_state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
         self.agents[i_agent].next_state = torch.from_numpy(next_state)
         self.net[self.agents[i_agent].group].memory_append((self.agents[i_agent].state, self.agents[i_agent].action, self.agents[i_agent].reward, self.agents[i_agent].next_state, self.done))

      for i_group in range(self.n_group):
         s_sample, a_sample, r_sample, next_s_sample, done_sample = self.net[i_group].memory_sample()
         # Optimize the model
         with torch.no_grad():
            target_q = self.net[i_group].target_q( next_s_sample ).detach().max(1)[0]
         target = r_sample + self.gamma*target_q*(1 - done_sample.float())
         main_q = self.net[i_group].forward( s_sample )
         val_t = main_q.gather(1, a_sample.unsqueeze(1).long() ).squeeze(1)
         loss = torF.mse_loss(val_t.float(), target.float(), reduction='mean')
         self.optimizer[i_group].zero_grad()
         loss.backward()
         nn.utils.clip_grad_norm_(self.net[i_group].main_net.parameters(), self.max_grad_norm)
         self.optimizer[i_group].step()
      
      if self.done:
         for i_group in range(self.n_group):
            self.net[i_group].target_net.load_state_dict(self.net[i_group].main_net.state_dict())
            if self.net[i_group].epsilon > self.epsilon_min:
               self.net[i_group].epsilon *= self.net[i_group].epsilon_decay
      
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
      self.total_reward = 0.0
      self.reward = 0.0
      self.group = 0

      self.state = np.zeros((self.hyperparameters.agent_eyesight_whole**2))
      self.next_state = np.zeros((self.hyperparameters.agent_eyesight_whole**2))

class Net(nn.Module):
   def __init__(self, hyperparameters):
      super().__init__()
      self.hyperparameters = hyperparameters
      self.state_size = 2*self.hyperparameters.agent_eyesight_whole**2
      self.action_size = 4
      self.layer_size = self.hyperparameters.layer_size
      self.fc1 = nn.Linear(self.state_size, self.layer_size)
      self.fc2 = nn.Linear(self.layer_size, self.action_size)


class WholeNet(nn.Module):
   def __init__(self, hyperparameters):
      super().__init__()
      self.hyperparameters = hyperparameters
      self.state_size = 2*self.hyperparameters.agent_eyesight_whole**2
      self.action_size = 4
      self.action_choice = [_ for _ in range (self.action_size)] 
      self.epsilon = self.hyperparameters.epsilon #epsilon-greedy
      self.epsilon_decay = self.hyperparameters.epsilon_decay

      self.main_net = Net(self.hyperparameters)
      self.target_net = copy.deepcopy(self.main_net)

      self.memory_size = self.hyperparameters.memory_size
      self.batch_size = self.hyperparameters.batch_size
      self.memory = deque([], maxlen=self.memory_size)

   def forward(self, x):
      x = x.float()
      u = torch.flatten(x, 1) # flatten all dimensions except batch
      u = torF.relu(self.main_net.fc1(u))
      q = self.main_net.fc2(u)
      return q

   def choose_action(self, q1):
      p = np.random.uniform(low=0.0,high=1.0)
      if p < self.epsilon:
         action_chosen = np.random.choice(self.action_choice) 
      else:
         action_chosen = torch.max(q1, dim=1)[1].detach().item() 
      return action_chosen

   def target_q(self, x):
      x = x.float()
      ut = torch.flatten(x, 1) # flatten all dimensions except batch
      ut = torF.relu(self.target_net.fc1(ut))
      qt = self.target_net.fc2(ut)
      return qt

   def memory_append(self, obj):
      self.memory.append(obj)

   def size(self):
      return len(self.memory)

   def memory_sample(self, device="cpu"):
      current_size = self.size()
      if current_size < self.batch_size:
         batch = random.sample(self.memory, current_size)
      else:
         batch = random.sample(self.memory, self.batch_size)
      res = []
      for i in range(5):
         k = np.stack(tuple(item[i] for item in batch), axis=0)
         res.append(torch.tensor(k, device=device))
      return res[0], res[1], res[2], res[3], res[4]


if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "_th_try.txt"
   filename_density = "density_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "_th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "_th_try.txt"
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
   filename_gif = "animation_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_" + str(seed) + "_th_try.gif"
   
   filename_snap1 = "snapshot_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_0_" + str(seed) + "_th_try.png"
   filename_snap2 = "snapshot_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_100_" + str(seed) + "_th_try.png"
   filename_snap3 = "snapshot_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_t_" + str(t_max-1) + "_" + str(seed) + "_th_try.png"
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
   filename_colormap = "colormap_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_group1_" + str(seed) + "_th_try.png"
   mean_view_white= (255.0*mean_view.transpose(1,0,2)).astype(np.uint8)
   mean_view_white[:,:,2] = np.where(mean_view_white[:,:,1] == 0, 255 - mean_view_white[:,:,0], 0)
   mean_view_white[:,:,1] = np.maximum( 255 - mean_view_white[:,:,0] , mean_view_white[:,:,1] )
   mean_view_white[:,:,0] = 255*np.where(mean_view_white[:,:,1] == mean_view_white[:,:,2], 1, 0)
   mean_Image = Image.fromarray( np.repeat(np.repeat(mean_view_white,5,axis=0),5,axis=1) )
   mean_Image.save(filename_colormap)

   filename_colormap = "colormap_DQN_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_memsize_" + str(hyper.memory_size) + "_init_distr_torchdefault_epsilon_" + str(hyper.epsilon_decay) + "to" + str(hyper.epsilon_min) + "_group2_" + str(seed) + "_th_try.png"
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
   