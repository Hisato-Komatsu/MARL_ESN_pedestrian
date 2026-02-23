"""
Source code for task.I
2024/9/28 random action agents used for the comparison of computational cost 
2026/1/16 improved the action choosing
"""
import numpy as np
import time
import copy
from PIL import Image

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
      self.width = 30
      self.height = 25
      self.n_agent = 1 #the number of agents
      self.agent_eyesight_whole = 11 #odd number


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

      self.build_wall()

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
      whole_action = self.agents[0].choose_action(self.n_agent)

      for i_agent in range(self.n_agent):

         self.agents[i_agent].action = whole_action[i_agent]
         self.agents[i_agent]._step(self.agents[i_agent].action)
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
            self.agents[i_agent].reward = self.agents[i_agent].reward_candidate
            self.agents[i_agent].pos = self.agents[i_agent].new_pos
         else:
            self.view[self.agents[i_agent].pos[0]][self.agents[i_agent].pos[1]][0] = self.agents[i_agent].color
            self.agents[i_agent].reward = 0.0

         self.agents[i_agent].total_reward += self.agents[i_agent].reward
         self.agents[i_agent].move_permission = 'no'

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
      self.action_size = 4

      self.total_reward = 0.0

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

   def choose_action(self, n): 
      action_chosen = np.random.randint(self.action_size, size=(n,))
      return action_chosen  # returns action only


if __name__ == '__main__':
   hyper = hyperparameters()
   t_max = hyper.t_max
   t_observe = hyper.t_observe
   ep_termination = hyper.ep_termination
   ep_observe = hyper.ep_observe

   filename_learning_curve = "LC_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_" + str(seed) + "th_try.txt"
   filename_density = "density_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_" + str(seed) + "th_try.txt"
   file_learning_curve = open(filename_learning_curve, "w")
   file_density = open(filename_density, "w")

   filename_time = "time_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try.txt"
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
   filename_gif = "animation_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try.gif"
   
   filename_snap1 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try_t_0.png"
   filename_snap2 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try_t_100.png"
   filename_snap3 = "snapshot_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_t_" + str(t_max-1) + "_" + str(seed) + "th_try.png"
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
   filename_colormap = "colormap_" + str(hyper.n_agent) + "pedestrians_task1_random_batch_randint_view_" + str(hyper.agent_eyesight_whole) + "_7_3_Nres_" + str(seed) + "th_try.png"
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
