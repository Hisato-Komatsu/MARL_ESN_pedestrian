"""
Source code for task.I
2024/9/28 random action agents used for the comparison of computational cost 
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
      self.width = 30
      self.height = 25
      self.n_agent = 1 #the number of agents
      self.agent_eyesight_whole = 11 #odd number

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
      self.width = self.hyperparameters.width
      self.height = self.hyperparameters.height
      self.area = self.width*self.height
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.pos_to_object = [[None for i in range(self.height)] for j in range(self.width)]
      self.n_agent = self.hyperparameters.n_agent
      self.done = False

      self.build_wall()

      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         if i_agent == 0:
            new_agent = Agent(pos_temp_agent[0], pos_temp_agent[1], self.condition, kind='agent', hyperparameters=self.hyperparameters)
         else:
            new_agent = copy.deepcopy(self.agents[0])
            new_agent.pos = pos_temp_agent
         self.agents.append(new_agent)
         self.pos_to_object[new_agent.pos[0]][new_agent.pos[1]] = new_agent
         self.condition[new_agent.pos[0]][new_agent.pos[1]] = condition_dict["agent"]
         self.view[new_agent.pos[0]][new_agent.pos[1]][0] = self.agents[i_agent].color

   def agent_position(self, i_agent):
      if i_agent <= 19 :
         return np.array([8+(i_agent%2)-2*(i_agent//4),7+(i_agent%4)])
      elif 20 <= i_agent <= 39 :
         return np.array([28+(i_agent%2)-2*((i_agent-20)//4),7+((i_agent-20)%4)])

   def build_wall(self):
      if self.walls: 
         for w1 in self.walls:
            self.pos_to_object[w1.pos[0]][w1.pos[1]] = w1
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
      new_wall = Object(x,y, self.condition, kind=("wall"), hyperparameters=self.hyperparameters)
      self.walls.append(new_wall)
      self.pos_to_object[x][y] = new_wall
      self.view[x][y][1] = new_wall.color

   def reset(self):
      #reset of environment with keeping the result of training
      self.condition = np.zeros((self.width,self.height), dtype=int).tolist()
      self.view = np.zeros((self.width,self.height,2))
      self.move_candidate_number = np.zeros((self.width,self.height)).astype(int)
      self.pos_to_object = [[None for i in range(self.height)] for j in range(self.width)]
      self.done = False
      self.build_wall()
      for i_agent in range(self.n_agent):
         pos_temp_agent = self.agent_position(i_agent)
         #self.agents[i_agent].X_esn = np.zeros(self.agents[i_agent].reservoir_size)
         self.agents[i_agent].pos = pos_temp_agent
         self.agents[i_agent].total_reward = 0.0
         self.pos_to_object[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = self.agents[i_agent]
         self.condition[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = condition_dict["agent"]
         self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color

   def _step(self, t):
      #agents' action
      for i_agent in range(self.n_agent):
         action=[0,0]
         #self.agents[i_agent]._key_step(self.pos_to_object)
         """
         #Neural net is not required for random action agents.
         if t >= 1:
            self.agents[i_agent].reward_past = self.agents[i_agent].reward
            self.agents[i_agent].action_past = action
            self.agents[i_agent].Xi_temp_part0 = copy.deepcopy(self.agents[i_agent].Xi_temp_part1)

         if self.agents[i_agent].dead_or_alive == 'alive':
            delx = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[0]
            dely = self.agents[i_agent].agent_eyesight_1side - self.agents[i_agent].pos[1]
            state = np.roll(np.roll(self.view, delx, axis=0), dely, axis=1)[0:self.agents[i_agent].agent_eyesight_whole, 0:self.agents[i_agent].agent_eyesight_whole, :]
         else:
            state = np.zeros( (self.agents[i_agent].agent_eyesight_whole,self.agents[i_agent].agent_eyesight_whole,2) )
         self.agents[i_agent].action, self.agents[i_agent].Xi_temp_part1, self.agents[i_agent].Q_temp = self.agents[i_agent].choose_action(state)
         """
         self.agents[i_agent].action = self.agents[i_agent].choose_action()
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
            self.pos_to_object[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]] = None
            self.pos_to_object[self.agents[i_agent].new_pos[0]][self.agents[i_agent].new_pos[1]] = self.agents[i_agent]
            self.agents[i_agent].reward = self.agents[i_agent].reward_candidate
            self.agents[i_agent].pos = self.agents[i_agent].new_pos
         else:
            self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            self.agents[i_agent].reward = 0.0

         self.agents[i_agent].total_reward += self.agents[i_agent].reward
         self.agents[i_agent].move_permission = 'no'

         """
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
         """

      """
      #Neural net is not required for random action agents.
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
      """

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
         view_3ch = np.concatenate( [self.view, np.zeros((self.width,self.height,1))], axis=2 ).transpose(1,0,2)
         view_3ch = np.maximum( 255.0*np.minimum( view_3ch, np.ones(view_3ch.shape) ), np.zeros(view_3ch.shape) ).astype(np.uint8)
         self.Image_list.append( Image.fromarray( np.repeat(np.repeat(view_3ch,5,axis=0),5,axis=1) ) )
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

class Agent(Object):
   def __init__(self,x,y,condition, kind, hyperparameters):
      super().__init__(x,y,condition, kind, hyperparameters)
      # building neural net
      #self.state_size = self.agent_eyesight_whole*self.agent_eyesight_whole*2
      self.action_size = 4
      self.action_choice = [_ for _ in range (self.action_size)] 

      self.total_reward = 0.0
      self.reward = 0.0
      self.eaten_foods = 0
      self.dead_or_alive = 'alive'


   """
   #methods related with reinforcement learning ... not required
   def _ReLU(self,x):
      zeros = np.zeros(x.shape)
      return np.maximum(x,zeros)

   def _softmax(self,z):
      z_rescale = self.inverse_temperature*(z - np.max(z))
      z_exp = np.exp(np.clip(z_rescale,-250,250))
      return z_exp/( np.sum(z_exp) )
   """

   def choose_action(self): #epsilon-greedy
      action_chosen = np.random.choice(self.action_choice) 
      return action_chosen  # returns action only

   """
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
   """



if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_" + str(seed) + "th_try.txt"
   filename_density = "density_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_" + str(seed) + "th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try.txt"
   file_time = open(filename_time, "w")

   t1 = 0
   myEnv = Env(hyperparameters=hyper)
   total_reward_list = []
   mean_view = np.zeros((myEnv.width,myEnv.height,2))
   var_view = np.zeros((myEnv.width,myEnv.height,2))
   mean_count = 0.0
   #print( myEnv.condition[myEnv.agents[0].pos[0]][myEnv.agents[0].pos[1]] )
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
   filename_gif = "animation_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try.gif"
   
   filename_snap1 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try_t_0.png"
   filename_snap2 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try_t_100.png"
   filename_snap3 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_t_" + str(t_max-1) + "_" + str(seed) + "th_try.png"
   myEnv._rendermode_change_to_rgb()
   myEnv.done = False
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
   filename_colormap = "colormap_" + str(hyper.n_agent) + "pedestrians_task1_random_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try.png"
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
