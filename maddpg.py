import torch as T
import torch.nn.functional as F
from agent import Agent
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    # This function modifies the largest value in a given list to 1 and the rest to 0
    def one_hot_encode(self, lst):
        arr = np.array(lst)
        max_idx = arr.argmax()
        one_hot = np.zeros_like(arr)
        one_hot[max_idx] = 1
        return one_hot.astype(int).tolist()
            
    # returns dict of each agents most likely action discretized E.G: [0,0,0,0,1] chooses action 5 
    # (I could handle this is the agent choose function not sure if that better)
    def choose_action(self, raw_obs):        
        actions = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            action_probablities = agent.choose_action(raw_obs[agent_id])
            best_action= max(range(len(action_probablities)), key=action_probablities.__getitem__)
            actions[agent_id] = best_action
            
        return actions

    # Adjusts actor and critic wieghts
    def learn(self, memory):
        # If memory is not the size of a batch size (1024) then return
        if not memory.ready():
            return
        
        #print('learn')

        # Samples algorithms central memory
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        # Makes sure each tensor is working on the same device, should be gpu (cuda:0)
        device = self.agents[0].actor.device
        
        """ print('Before modifications:')
        print('states:', states)
        print('actions:', actions)
        print('rewards:', rewards)
        print('states_:', states_)
        print('dones:', dones) """
        
        # converts sampled memory list of arrays to Tensors
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        
        """ print('After modifications:')
        print('states:', states)
        print('actions:', actions)
        print('rewards:', rewards)
        print('states_:', states_)
        print('dones:', dones) """


        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        # self.agent return a list of the each agent created from Agent class
        # Below creates tensors from each agents seperate actor memory
        for agent_idx, agent in enumerate(self.agents):
            # Creates tensor from next observation of individual agents sampled from their memory
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            # Gets action probabilities given agent individual next observation 
            new_pi = agent.target_actor.forward(new_states).to(device)
            # Stores actions
            all_agents_new_actions.append(new_pi)
            # Creates tensor from current observation of individual agents sampled from their memory
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            # Gets action probabilities given inidiviaul agent current observation
            pi = agent.actor.forward(mu_states).to(device)
            # Stores actions
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        # Concatonates tensors stores in all_agents_new_actions which is the actions choosen by individial agents given there individual next observations
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        # Same as above but instead from individual agents curent observation
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        # The actions sampled from the individual memory
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # Gets Q_values from state action pair for each agent given its next state and next action it takes
            critic_value_ = agent.target_critic.forward(states_, new_actions)
            # Gets Q_values from state action pair for each agent given its current state and the action it took in it
            critic_value = agent.critic.forward(states, old_actions)        
            # sets value to 0.0 if state is done
            critic_value_[dones[:,0]] = 0.0
            critic_value_ = critic_value_.view(-1)
            
            
            #print('critic_value: ', critic_value)

            # Calculates target value based on algorithm formula 
            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            
            #make target the depth of the batch size
            target = target.view(1024,1)
            
            
            #print('target_shape', target.shape)
            #print('critic_shpae:', critic_value.shape)
            #target = target.view(-1, 1)            

            # The order of zero_grad, loss_func and backward call could be wrong. I dont fuilly understand the intuition in terms of order
            # order was different from normal DDPG and MADDPG (Currently following DDPG order from course)
            agent.critic.optimizer.zero_grad()
            # Calculates the loss between the  target and critic_value
            critic_loss = F.mse_loss(target, critic_value)
            # optimize (honestly dont know where this ties in with paper alg)
            #agent.critic.optimizer.zero_grad()
            #print('critic.loss', critic_loss)
            #print('crit loss verison: ', critic_loss._version)
            # Backwards propagate
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # Changed the minus sign from -T.mean() to -agent.critic....
            agent.actor.optimizer.zero_grad()
            # Dont even know what im looking at but it returns action state value (Q-value)
            actor_loss = -agent.critic.forward(states, mu).flatten()
            # Dont know what this means
            actor_loss = T.mean(actor_loss)
            #print('verrrrsion:', actor_loss._version)
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            #print('checkkkkkkkkkk')
            # Updates the parameters probably some sort of gradient decent
            agent.update_network_parameters()
            #print('mi remember')
            
            # code makes it through learn function once before getting caught at the first backward called on critic_loss.
            # Alot of people saying to remove retain_graph but im pretty sure we need it for MADDPG
