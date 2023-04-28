import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
#from make_env import make_env
from pettingzoo.mpe import simple_adversary_v2
import torch as T
import gc

def obs_list_to_state_vector(observation):
    state = np.array([])
    #print('observation: ', observation)
    
    for obs in observation:
        #print('obs: ', obs)
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    #scenario = 'simple_adversary'
    #env = simple_adversary_v2.env()
    #env.reset()
    #n_agents = len(env.agents)
    #list_of_agents = env.agents
    #actor_dims = []
    
    T.cuda.empty_cache()
    gc.collect()

    parallel_env = simple_adversary_v2.parallel_env()
    scenario = 'simple_adversary'
    # I only do this outside loop because to access certain properties we need to activate the env
    initial_temp = parallel_env.reset()
    n_agents = parallel_env.num_agents
    agents = parallel_env.agents
    
    actor_dims = []
    
    # Actor_dims.append(env.observation_space[i].shape[0])
    for agent in parallel_env.agents:
        actor_dims.append(parallel_env.observation_space(agent).shape[0])

    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    # Since we assume each agent has the same action space we just take the action space of the first agent

    n_actions = parallel_env.action_space(agents[0]).n
    #n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()
    for i in range(N_GAMES):
        obs = parallel_env.reset()
        # Convert dict -> list because thats what our algorithm uses
        list_obs = list(obs.values())

        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                parallel_env.render()
                #time.sleep(0.1) # to slow down the action for the video
            
            # convet obs to list instead of dict 
            actions = maddpg_agents.choose_action(obs)
            #print('actions: ', actions)
            #print('env action_space: ', parallel_env.action_spaces)
            obs_, reward, done, truncated, info = parallel_env.step(actions)
            # Convert dict -> list because thats what our algorithm uses
            list_done = list(done.values())
            list_reward = list(reward.values())
            list_actions = list(actions.values())
            list_obs_ = list(obs_.values())
            state = obs_list_to_state_vector(list_obs)
            state_ = obs_list_to_state_vector(list_obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(list_obs, state, list_actions, list_reward, list_obs_, state_, list_done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            #print('score: ', score, 'reward: ', reward)
            score += sum(list_reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
