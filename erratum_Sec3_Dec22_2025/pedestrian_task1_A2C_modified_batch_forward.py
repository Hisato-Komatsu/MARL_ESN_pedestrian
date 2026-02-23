"""
Source code for task.I using A2C
2025/9/29 improved the forward propagation to save the computational time
2025/10/8 employed logits instead of probs in choose_action
2025/12/26 changed the advantage trajectory into ndarray
2025/1/5 introduced next_state calculation
2026/1/6 introduced inference_mode()
2026/1/6 use inference_mode during exploring
2026/1/13 introduced "return_action" to the forward propagation
"""
import numpy as np
import time
import copy
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as torF
from torch.distributions import Categorical

start_time = time.time()
#np.set_printoptions(precision=3) # used for debugging

seed = 1
if torch.cuda.is_available():
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

global condition_dict
condition_dict = {"vacant":0, "wall":1, "agent":2}

class hyperparameters:
   def __init__(self):
      self.t_max = 500
      self.t_observe = 100
      self.ep_termination = 250
      self.ep_observe = 150
      self.width = 30
      self.height = 25
      self.n_agent = 1 #the number of agents
      self.agent_eyesight_whole = 11 #odd number

      self.gamma = 0.95
      self.layer_size = 1024

      self.learning_rate = 0.00025
      self.num_step = 5
      self.entropy_coef = 0.01
      self.value_loss_coef = 0.5
      self.max_grad_norm = 50.0


class Env:
   def __init__(self, hyperparameters, render_mode='off'):
      #render_mode : 'off' or 'rgb_array'
      self.render_mode = render_mode
      if self.render_mode == 'rgb_array' :
         self.Image_list = []

      self.agents = []
      self.walls = []
      self.hyperparameters = hyperparameters
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.n_agent = self.hyperparameters.n_agent
      self.done = False
      
      self.gamma = self.hyperparameters.gamma
      self.learning_rate = self.hyperparameters.learning_rate
      self.num_step = self.hyperparameters.num_step
      self.entropy_coef = self.hyperparameters.entropy_coef
      self.value_loss_coef = self.hyperparameters.value_loss_coef
      self.max_grad_norm = self.hyperparameters.max_grad_norm

      self.build_wall()

      self.net = Net(hyperparameters=self.hyperparameters)
      self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

      self.whole_mem_states = []
      self.whole_mem_actions = []
      self.whole_R_values = []

      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         if i_agent == 0:
            new_agent = Agent(pos_temp_agent[0], pos_temp_agent[1], self.condition, kind='agent', hyperparameters=self.hyperparameters)
         else:
            new_agent = copy.deepcopy(self.agents[0])
            new_agent.pos = pos_temp_agent
         self.agents.append(new_agent)
         self.condition[new_agent.pos[0]][new_agent.pos[1]] = condition_dict["agent"]
         self.view[new_agent.pos[0]][new_agent.pos[1]][0] = self.agents[i_agent].color

      self.agents[0].make_eyesight()

   def agent_position(self, i_agent):
      if i_agent <= 19 :
         return np.array([8+(i_agent%2)-2*(i_agent//4),7+(i_agent%4)])
      elif 20 <= i_agent <= 39 :
         return np.array([28+(i_agent%2)-2*((i_agent-20)//4),7+((i_agent-20)%4)])

   def build_wall(self):
      if self.walls: 
         for w1 in self.walls:
            self.view[w1.pos[0]][w1.pos[1]][1] = w1.color
            self.condition[w1.pos[0]][w1.pos[1]] = condition_dict["wall"]
      else:
         for x in range(self.width):
            for y in range(self.height):
               if y == 5 or y == 6:
                  self.set_wall(x,y)
               elif (y == 11 or y == 12) and (x<=6 or x>=23):
                  self.set_wall(x,y)
               elif (y == 19 or y == 20) and (x>=5 and x<=24):
                  self.set_wall(x,y)
               elif (y == 8 or y == 9 or y == 13 or y == 14) and (x>=11 and x<=18):
                  self.set_wall(x,y)
               elif (x == 11 or x == 12 or x == 17 or x == 18) and (y>=10 and y<=12):
                  self.set_wall(x,y)
               elif (x == 5 or x == 6 or x == 23 or x == 24) and (y>=13 and y<=18):
                  self.set_wall(x,y)

   def set_wall(self,x,y):
      new_wall = Object(x,y, self.condition, kind=("wall"))
      self.walls.append(new_wall)
      self.view[x][y][1] = new_wall.color

   def reset(self):
      #reset of environment with keeping the result of training
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.done = False
      self.build_wall()
      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         self.agents[i_agent].pos = pos_temp_agent
         self.agents[i_agent].total_reward = 0.0
         self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = condition_dict["agent"]
         self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color


   def _step(self, t):
      #agents' action
      if t == 0:
         # getting batched observation 
         whole_pos_x = np.array([agent.pos[0] for agent in self.agents]) 
         whole_pos_y = np.array([agent.pos[1] for agent in self.agents]) 
         x_id_batch = (whole_pos_x[:, None] + self.agents[0].offset[None, :]) % self.width
         y_id_batch = (whole_pos_y[:, None] + self.agents[0].offset[None, :]) % self.height
         state_tensor = torch.from_numpy(self.view[x_id_batch[:, :, None], y_id_batch[:, None, :]]).float().flatten(1) # preprocess the observtion previously
      else:
         state_tensor = self.next_state_tensor

      with torch.inference_mode():
         whole_action = self.net.forward( state_tensor, return_action=True ) 

      for i_agent in range(self.n_agent):
         self.agents[i_agent].action = whole_action[i_agent:i_agent+1]
         self.agents[i_agent]._step(self.agents[i_agent].action.item())
         self.move_candidate_number[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] += 1            

         self.agents[i_agent].memory_states.append(state_tensor[i_agent])
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
            self.agents[i_agent].reward = self.agents[i_agent].reward_candidate
            self.agents[i_agent].pos = self.agents[i_agent].new_pos
         else:
            self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            self.agents[i_agent].reward = 0.0

         self.agents[i_agent].total_reward += self.agents[i_agent].reward
         self.agents[i_agent].move_permission = 'no'

         self.agents[i_agent].memory_rewards.append(self.agents[i_agent].reward)
         self.agents[i_agent].memory_dones.append(self.done)

      # getting batched observation 
      whole_pos_x = np.array([agent.pos[0] for agent in self.agents]) 
      whole_pos_y = np.array([agent.pos[1] for agent in self.agents]) 
      x_id_batch = (whole_pos_x[:, None] + self.agents[0].offset[None, :]) % self.width
      y_id_batch = (whole_pos_y[:, None] + self.agents[0].offset[None, :]) % self.height
      self.next_state_tensor = torch.from_numpy(self.view[x_id_batch[:, :, None], y_id_batch[:, None, :]]).float().flatten(1)

      if self.done or (t+1)%self.num_step ==0 :
         with torch.inference_mode():
            whole_R_temp = self.net.forward( self.next_state_tensor, return_action=False ).tolist() 

         for i_agent in range(self.n_agent):
            R_temp = whole_R_temp[i_agent]
            len_mem = len(self.agents[i_agent].memory_rewards)
            self.agents[i_agent].R_values = np.zeros( (len_mem, 1) )

            for i_rev_step in reversed( range(len_mem) ):
               R_temp = self.agents[i_agent].memory_rewards[i_rev_step] + self.gamma * R_temp*(1.0-self.agents[i_agent].memory_dones[i_rev_step])
               self.agents[i_agent].R_values[i_rev_step] = R_temp

            self.whole_mem_states += self.agents[i_agent].memory_states
            self.whole_mem_actions += self.agents[i_agent].memory_actions
            self.whole_R_values.append(torch.from_numpy(self.agents[i_agent].R_values))

         tensor_mem_states = torch.stack(self.whole_mem_states)
         tensor_mem_actions = torch.stack(self.whole_mem_actions)
         tensor_mem_Rs = torch.cat( self.whole_R_values, dim=0).float()

         log_prob_new, entropy_new, value_new = self.net.new_probs(tensor_mem_states, tensor_mem_actions)   
         self.whole_advantage = (tensor_mem_Rs - value_new)

         self.value_loss = torch.mean( self.whole_advantage.pow(2) ) 
         self.policy_loss = torch.mean( -log_prob_new*self.whole_advantage.detach() - self.entropy_coef * entropy_new )

         self.optimizer.zero_grad()
         (self.policy_loss + self.value_loss_coef * self.value_loss).backward()
         nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
         self.optimizer.step()

         self.whole_mem_states = []
         self.whole_mem_actions = []
         self.whole_R_values = []

         for i_agent in range(self.n_agent):
            self.agents[i_agent].memory_clear()

      #resetting move_candidate_number
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)

   def _render(self):
      if self.render_mode == 'rgb_array' :
         view_3ch = np.concatenate( [self.view, np.zeros((self.width,self.height,1))], axis=2 ).transpose(1,0,2)
         view_3ch = np.maximum( 255.0*np.minimum( view_3ch, np.ones(view_3ch.shape) ), np.zeros(view_3ch.shape) ).astype(np.uint8)
         self.Image_list.append( Image.fromarray( np.repeat(np.repeat(view_3ch,5,axis=0),5,axis=1) ) )

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

      self.memory_states = []
      self.memory_actions = []
      self.memory_rewards = []
      self.memory_dones = []

   def make_eyesight(self): # lazy initialization for particular agents' attribute
      self.agent_eyesight_whole = self.hyperparameters.agent_eyesight_whole 
      self.agent_eyesight_1side = self.agent_eyesight_whole//2
      self.offset = np.arange(-self.agent_eyesight_1side, self.agent_eyesight_1side + 1)

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
      self.reward_candidate = float(direction[0])
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
    
   def memory_clear(self):
      self.memory_states = []
      self.memory_actions = []
      self.memory_rewards = []
      self.memory_dones = []   


class Net(nn.Module):
   def __init__(self, hyperparameters):
      super().__init__()
      self.hyperparameters = hyperparameters
      self.state_size = 2*self.hyperparameters.agent_eyesight_whole**2
      self.action_size = 4
      self.layer_size = self.hyperparameters.layer_size

      self.fc1 = nn.Linear(self.state_size, self.layer_size)
      self.fc2_policy = nn.Linear(self.layer_size, self.action_size)
      self.fc2_value = nn.Linear(self.layer_size, 1)

   def forward(self, x, return_action):
      u = torF.relu(self.fc1(x))
      if return_action:
         logits = self.fc2_policy(u)
         probs = Categorical(logits=logits)
         action = probs.sample()
         return action
      else:
         value = self.fc2_value(u)
         return torch.squeeze(value,1)

   def new_probs(self, x, a):
      u = torF.relu(self.fc1(x))
      logits = self.fc2_policy(u)
      value = self.fc2_value(u)
      probs = Categorical(logits=logits)
      log_prob = probs.log_prob(a.long().squeeze(-1))
      return torch.unsqueeze(log_prob,dim=1), torch.unsqueeze(probs.entropy(),dim=1), value


if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.txt"
   filename_density = "density_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.txt"
   file_time = open(filename_time, "w")

   t1 = 0
   myEnv = Env(hyperparameters=hyper)
   total_reward_list = []
   mean_view = np.zeros((myEnv.width,myEnv.height,2))
   var_view = np.zeros((myEnv.width,myEnv.height,2))
   mean_count = 0.0
   for n_episode in range (ep_termination):
      myEnv.done = False
      for t1 in range (t_max):
         myEnv._render()
         if t1 == t_max-1:
            myEnv.done = True
         if n_episode >= ep_observe and t1 >= t_observe:
            mean_view += myEnv.view
            var_view += myEnv.view*myEnv.view
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
         print(i,myEnv.height - 1 - j, mean_view[i,j,0], err_view[i,j,0], mean_view[i,j,1], err_view[i,j,1], sep="	", file=file_density )
      print("", file=file_density)

   file_learning_curve.close()
   file_density.close()
   
   #gif-animation
   filename_gif = "animation_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.gif"
   
   filename_snap1 = "snapshot_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_t_0_" + str(seed) + "th_try.png"
   filename_snap2 = "snapshot_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_t_100_" + str(seed) + "th_try.png"
   filename_snap3 = "snapshot_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_t_" + str(t_max-1) + "_" + str(seed) + "th_try.png"
   myEnv._rendermode_change_to_rgb()
   myEnv.done = False
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
   filename_colormap = "colormap_A2C_logit_bf_stack_nogr_infe_" + str(hyper.n_agent) + "pedestrians_task1_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(hyper.layer_size) + "_lr_" + str(hyper.learning_rate) + "_numstep_" + str(hyper.num_step) + "_" + str(seed) + "th_try.png"
   mean_view_3ch = np.concatenate( [mean_view, np.zeros((myEnv.width,myEnv.height,1))], axis=2 ).transpose(1,0,2)
   mean_view_3ch = (255.0*mean_view_3ch).astype(np.uint8)
   mean_view_3ch[:,:,2] = np.where(mean_view_3ch[:,:,1] == 0, 255 - mean_view_3ch[:,:,0], 0)
   mean_view_3ch[:,:,1] = np.maximum( 255 - mean_view_3ch[:,:,0] , mean_view_3ch[:,:,1] )
   mean_view_3ch[:,:,0] = 255*np.where(mean_view_3ch[:,:,1] == mean_view_3ch[:,:,2], 1, 0)
   mean_Image = Image.fromarray( np.repeat(np.repeat(mean_view_3ch,5,axis=0),5,axis=1) )
   mean_Image.save(filename_colormap)

   print( np.mean( np.array(total_reward_list) ) )
   for i1 in total_reward_list:
      print(i1)

   print(time.time() - start_time, file=file_time)
