from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt
import torch

from agent import Agent

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", no_graphics=False)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    print("training the DDPG model")

    agent1 = Agent(state_size=state_size * 2, action_size=action_size)
    agent2 = Agent(state_size=state_size * 2, action_size=action_size)
    
    agent1.online_actor.load_state_dict(torch.load('actor_model1.path'))
    agent1.online_critic.load_state_dict(torch.load('critic_model1.path'))
    
    agent2.online_actor.load_state_dict(torch.load('actor_model2.path'))
    agent2.online_critic.load_state_dict(torch.load('critic_model2.path'))
    
    num_episodes = 10
    
    agent_scores = np.zeros(num_agents)
    
    for episode in range(1, num_episodes+1):
        
        env_info = env.reset(train_mode=False)[brain_name]
        
        state = env_info.vector_observations
        state = np.reshape(state,(1,48))
        
        agent1.reset()
        agent2.reset()
        
        
        
        while True:
            
            action1 = agent1.act(state, add_noise=False)
            action2 = agent2.act(state, add_noise=False)
            
            action =[action1, action2]            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            next_state = np.reshape(next_state, (1, -1))
            rewards = env_info.rewards
            dones = env_info.local_done

            agent_scores += rewards
            
            state = next_state
            
            if np.any(dones):
                break
        
        
        print("\r Agent 1: {} x {} : Agent 2".format(agent_scores[0], agent_scores[1]), end="", flush=True)
              
print("Agent {} wins with {} points".format(np.argmax(agent_scores), np.max(agent_scores)))
                
            
            
            
                           
        


   