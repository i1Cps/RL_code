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
            
    # Loop though all agents and pass the obs value from raw_obs into choose_action function, which returns an action
    def choose_action(self, raw_obs):        
        actions = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            # choose_action returns 5 probalities, we need to return the index of the highest one
            # highest probablity, set it to 1 and set the rest to 0
            # example agent_actions :[0.27671874, 0.69834113, 0.9941813 , 0.45730233, 0.48765135]
            action_probablities = agent.choose_action(raw_obs[agent_id])
            best_action= max(range(len(action_probablities)), key=action_probablities.__getitem__)
            actions[agent_id] = best_action
            
        return actions

    def learn(self, memory):
        if not memory.ready():
            return
        print('learn')

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device
        
        print('Before modifications:')
        print('states:', states)
        print('actions:', actions)
        print('rewards:', rewards)
        print('states_:', states_)
        print('dones:', dones)

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        
        print('After modifications:')
        print('states:', states)
        print('actions:', actions)
        print('rewards:', rewards)
        print('states_:', states_)
        print('dones:', dones)


        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        # self.agent return a list of the each agent object created for the environment       
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            #critic_value_ = agent.target_critic.forward(states_, new_actions).flatten().clone()

            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()
            #print('critic_value: ', critic_value)

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            print('target_shape', target.shape)
            print('critic_shpae:', critic_value.shape)
            #            target = target.view(-1, 1)            


            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            #print('critic.loss', critic_loss)
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            print('checkkkkkkkkkk')
            # pretty sure this function is causing all the mess never mind anything in this while loop coudl cause it
            agent.update_network_parameters()
            print('mi remember')
