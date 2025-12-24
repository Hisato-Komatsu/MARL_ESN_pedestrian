"""
Source code for task.II
2025 8/2 abolished replay buffer
"""
import numpy as np
import time
import os
import copy
from PIL import Image, ImageDraw
from collections import namedtuple
from collections import deque

import torch
from torch import nn
import torch.nn.functional as torF
from torch.distributions import Categorical

start_time = time.time()
np.set_printoptions(precision=3)

seed = 1
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
      self.n_agent = 48 #the number of agents
      self.agent_eyesight_whole = 11 #odd number

      self.gamma = 0.95
      self.layer_size = 1024

      self.n_group = 2 #number of groups of agents

      self.learning_rate = 0.00005
      self.num_step = 500
      self.num_epochs = 4
      self.minibatch_size = 125
      self.entropy_coef = 0.01
      self.value_loss_coef = 0.5
      self.gae_lambda = 0.95
      self.epsilon_clip = 0.2
      self.max_grad_norm = 1.0


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
      self.minibatch_size = self.hyperparameters.minibatch_size
      self.done = False
      
      self.gamma = self.hyperparameters.gamma
      self.learning_rate = self.hyperparameters.learning_rate
      self.num_step = self.hyperparameters.num_step
      self.num_epochs = self.hyperparameters.num_epochs
      self.entropy_coef = self.hyperparameters.entropy_coef
      self.value_loss_coef = self.hyperparameters.value_loss_coef
      self.gae_lambda = self.hyperparameters.gae_lambda
      self.epsilon_clip = self.hyperparameters.epsilon_clip
      self.max_grad_norm = self.hyperparameters.max_grad_norm

      self.build_wall()

      self.net = [_ for _ in range(self.n_group)]
      self.optimizer = [_ for _ in range(self.n_group)]
      self.policy_loss = [_ for _ in range(self.n_group)]
      self.value_loss = [_ for _ in range(self.n_group)]
      self.whole_mem_states = [_ for _ in range(self.n_group)]
      self.whole_mem_actions = [_ for _ in range(self.n_group)]
      self.whole_R_values = [_ for _ in range(self.n_group)]
      self.whole_GAE_values = [_ for _ in range(self.n_group)]
      self.whole_mem_Rs = [_ for _ in range(self.n_group)]
      self.whole_mem_GAEs = [_ for _ in range(self.n_group)]
      self.whole_mem_probs = [_ for _ in range(self.n_group)]
      self.whole_advantage = [_ for _ in range(self.n_group)]
      for i_group in range(self.n_group):
         self.net[i_group] = Net(hyperparameters=self.hyperparameters)
         self.optimizer[i_group] = torch.optim.Adam(self.net[i_group].parameters(), lr=self.learning_rate)
         self.policy_loss[i_group] = 0.0
         self.value_loss[i_group] = 0.0

         self.whole_mem_states[i_group] = []
         self.whole_mem_actions[i_group] = []
         self.whole_R_values[i_group] = np.zeros( (0,1) )
         self.whole_GAE_values[i_group] = np.zeros( (0,1) )
         self.whole_mem_probs[i_group] = []

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
         
         delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
         dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
         state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
         state = torch.from_numpy(state)
         with torch.no_grad():
            self.agents[i_agent].action, self.agents[i_agent].prob, self.agents[i_agent].entropy, self.agents[i_agent].value = self.net[self.agents[i_agent].group].forward( state[None, :] )
         self.agents[i_agent]._step(self.agents[i_agent].action.item(), self.pos_to_object)
         self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] += 1            
         
         self.agents[i_agent].memory_states.append(state)
         self.agents[i_agent].memory_actions.append(self.agents[i_agent].action)

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

         #update of memory matrices...deleted

         self.agents[i_agent].memory_values.append(self.agents[i_agent].value)
         self.agents[i_agent].memory_probs.append(self.agents[i_agent].prob)
         self.agents[i_agent].memory_rewards.append(self.agents[i_agent].reward)
         self.agents[i_agent].memory_dones.append(self.done)

      if self.done or (t+1)%self.num_step ==0 :
         for i_agent in range(self.n_agent):
            #calculation on the last step
            delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
            dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
            state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
            state = torch.from_numpy(state)

            with torch.no_grad() :
               _, _, _, V_last = self.net[self.agents[i_agent].group].forward( state[None, :] )
            V_last = V_last.detach().item()
            self.agents[i_agent].GAE_values = np.zeros( (len(self.agents[i_agent].memory_values),1) )

            memvalue_numpy = torch.tensor(self.agents[i_agent].memory_values).detach().numpy()
            for i_rev_step in reversed( range(len(self.agents[i_agent].memory_values)) ):
               if i_rev_step == len(self.agents[i_agent].memory_values) - 1 :
                  self.agents[i_agent].GAE_values[i_rev_step] = self.agents[i_agent].memory_rewards[i_rev_step] + self.gamma*V_last*(1.0-self.agents[i_agent].memory_dones[i_rev_step]) - memvalue_numpy[i_rev_step]
               else:
                  delta_t = self.agents[i_agent].memory_rewards[i_rev_step] + self.gamma*memvalue_numpy[i_rev_step + 1] - memvalue_numpy[i_rev_step]
                  self.agents[i_agent].GAE_values[i_rev_step] = self.agents[i_agent].GAE_values[i_rev_step + 1]*self.gamma*self.gae_lambda + delta_t
            self.agents[i_agent].R_values = self.agents[i_agent].GAE_values + memvalue_numpy.reshape(-1,1)

            self.whole_mem_states[self.agents[i_agent].group] += self.agents[i_agent].memory_states
            self.whole_mem_actions[self.agents[i_agent].group] += self.agents[i_agent].memory_actions
            self.whole_R_values[self.agents[i_agent].group] = np.concatenate((self.whole_R_values[self.agents[i_agent].group], self.agents[i_agent].R_values),axis=0)
            self.whole_GAE_values[self.agents[i_agent].group] = np.concatenate((self.whole_GAE_values[self.agents[i_agent].group], self.agents[i_agent].GAE_values),axis=0)
            self.whole_mem_probs[self.agents[i_agent].group] += self.agents[i_agent].memory_probs

         #trainig
         for i_group in range(self.n_group):
            batch_index = np.arange(len(self.whole_mem_actions[i_group]))
            tensor_mem_probs = torch.stack(list(self.whole_mem_probs[i_group]))
            tensor_mem_states = torch.stack(list(self.whole_mem_states[i_group]))
            tensor_mem_actions = torch.stack(list(self.whole_mem_actions[i_group]))
            tensor_mem_Rs = torch.tensor( self.whole_R_values[i_group] )
            tensor_mem_GAEs = torch.tensor( self.whole_GAE_values[i_group] )

            for epoch in range(self.num_epochs) :
               np.random.shuffle(batch_index) 
               for start_id in range(0, len(self.whole_mem_actions[i_group]), self.minibatch_size):
                  end_id = min(start_id + self.minibatch_size, len(self.whole_mem_actions[i_group]) )
                  minibatch_index = batch_index[start_id:end_id]

                  probs_old = tensor_mem_probs[minibatch_index].detach()
                  replay_states = tensor_mem_states[minibatch_index]
                  replay_actions = tensor_mem_actions[minibatch_index]
                  R_sample = tensor_mem_Rs[minibatch_index]
                  probs_new, entropy_new, value_new = self.net[i_group].new_probs(replay_states,replay_actions)
                  ratio = probs_new/probs_old
                  clipped_ratio = torch.clip( ratio, 1.0-self.epsilon_clip, 1.0+self.epsilon_clip)

                  gae_sample = tensor_mem_GAEs[minibatch_index]
                  policy_loss1 = ratio*gae_sample
                  policy_loss2 = clipped_ratio*gae_sample

                  self.value_loss[i_group] = torch.mean( (value_new - R_sample)**2 )  
                  self.policy_loss[i_group] = torch.mean( - torch.min(policy_loss1,policy_loss2) - self.entropy_coef * entropy_new )

                  self.optimizer[i_group].zero_grad()
                  (self.policy_loss[i_group] + self.value_loss_coef * self.value_loss[i_group]).backward()
                  nn.utils.clip_grad_norm_(self.net[i_group].parameters(), self.max_grad_norm)
                  self.optimizer[i_group].step()


            self.whole_mem_states[i_group] = []
            self.whole_mem_actions[i_group] = []
            self.whole_R_values[i_group] = np.zeros( (0,1) )
            self.whole_GAE_values[i_group] = np.zeros( (0,1) )
            self.whole_mem_probs[i_group] = []

         for i_agent in range(self.n_agent):
            self.agents[i_agent].memory_clear()

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

      self.memory_probs = []
      self.memory_states = []
      self.memory_actions = []
      self.memory_values = []
      self.memory_rewards = []
      self.memory_dones = []
      self.R_values = np.zeros((1, 1))
      self.GAE_values = np.zeros((1, 1))

      self.value = torch.zeros(1, 1)
      self.entropy = torch.zeros(1, 1)
      self.prob = torch.zeros(1, 1)
      self.R = torch.zeros(1, 1)
      self.advantage = torch.zeros(1, 1)
    
   def memory_clear(self):
      self.memory_probs.clear()
      self.memory_states.clear()
      self.memory_actions.clear()
      self.memory_values.clear()
      self.memory_rewards.clear()
      self.memory_dones.clear()  

   def _zip(self):
      return zip(self.memory_probs, self.memory_values, self.memory_rewards, self.memory_dones)

   def load_data(self):
      for data in self._zip():
         return data
    
   def memory_reversed(self):
      for data in list(self._zip())[::-1]:
         yield data

class Net(nn.Module):
   def __init__(self, hyperparameters):
      super().__init__()
      self.hyperparameters = hyperparameters
      self.state_size = 2*self.hyperparameters.agent_eyesight_whole**2
      self.action_size = 4
      self.action_choice = [_ for _ in range (self.action_size)] 
      self.layer_size = self.hyperparameters.layer_size

      self.fc1 = nn.Linear(self.state_size, self.layer_size)
      self.fc2_policy = nn.Linear(self.layer_size, self.action_size)
      self.fc2_value = nn.Linear(self.layer_size, 1)


   def forward(self, x):
      x = x.float()
      u = torch.flatten(x, 1) # flatten all dimensions except batch
      u = torF.relu(self.fc1(u))
      policy = torF.softmax(self.fc2_policy(u), dim=1)
      value = self.fc2_value(u)
      probs = Categorical(probs=policy)
      action = probs.sample()
      return action, torch.squeeze(probs.probs[:,action],1), probs.entropy(), torch.squeeze(value,1)


   def new_probs(self, x, a):
      x = x.float()
      u = torch.flatten(x, 1) # flatten all dimensions except batch
      u = torF.relu(self.fc1(u))
      policy = torF.softmax(self.fc2_policy(u), dim=1)
      value = self.fc2_value(u)
      probs = Categorical(probs=policy)
      return probs.probs.gather(1, a.long() ), torch.unsqueeze(probs.entropy(),dim=1), value


if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.txt"
   filename_density = "density_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.txt"
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
   filename_gif = "animation_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.gif"
   
   filename_snap1 = "snapshot_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_t_0_" + str(seed) + "th_try.png"
   filename_snap2 = "snapshot_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_t_100_" + str(seed) + "th_try.png"
   filename_snap3 = "snapshot_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_t_" + str(t_max-1) + "_" + str(seed) + "th_try.png"
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
   filename_colormap = "colormap_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_group1_" + str(seed) + "th_try.png"
   mean_view_white= (255.0*mean_view.transpose(1,0,2)).astype(np.uint8)
   mean_view_white[:,:,2] = np.where(mean_view_white[:,:,1] == 0, 255 - mean_view_white[:,:,0], 0)
   mean_view_white[:,:,1] = np.maximum( 255 - mean_view_white[:,:,0] , mean_view_white[:,:,1] )
   mean_view_white[:,:,0] = 255*np.where(mean_view_white[:,:,1] == mean_view_white[:,:,2], 1, 0)
   mean_Image = Image.fromarray( np.repeat(np.repeat(mean_view_white,5,axis=0),5,axis=1) )
   mean_Image.save(filename_colormap)

   filename_colormap = "colormap_PPO_" + str(hyper.n_agent) + "pedestrians_task2_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_group2_" + str(seed) + "th_try.png"
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
   