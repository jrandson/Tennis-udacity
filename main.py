from unityagents import UnityEnvironment
import numpy as np

import matplotlib.pyplot as plt
import torch

from agent import Agent


def run_ddpg(env, agent1, agent2, brain_name, max_episodes=1000, max_steps=10000):
    scores = []

    for episode in range(1, max_episodes + 1):
        agent1.reset()
        agent1.reset()
        episode_score = np.zeros(2)

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        state = np.reshape(state, (1, -1))
        for step in range(max_steps):

            action1 = agent1.act(state, add_noise=True)
            action2 = agent2.act(state, add_noise=True)

            action = [action1, action2]
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            next_state = np.reshape(next_state, (1, -1))
            rewards = env_info.rewards
            dones = env_info.local_done

            episode_score += rewards

            agent1.step(state, action1, rewards[0], next_state, dones[0])
            agent2.step(state, action2, rewards[1], next_state, dones[1])

            state = next_state

            if np.any(dones):
                break

        scores.append(np.max(episode_score))
        mean_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            episode, mean_score, np.max(episode_score)), end="", flush=True)

        if mean_score >= 0.5:
            print("\t Model reached the score goal in {} episodes!".format(episode))
            break

    torch.save(agent1.online_actor.state_dict(), "actor_model1.path")
    torch.save(agent1.online_critic.state_dict(), "critic_model1.path")

    torch.save(agent2.online_actor.state_dict(), "actor_model2.path")
    torch.save(agent2.online_critic.state_dict(), "critic_model2.path")

    return scores


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
    scores = run_ddpg(env, agent1, agent2, brain_name, max_episodes=5000, max_steps=1000)


    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Episode #')
    fig.savefig("score_x_apisodes.png")
    plt.show()

    w = 10
    mean_score = [np.mean(scores[i - w:i]) for i in range(w, len(scores))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(mean_score) + 1), mean_score)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Episode #')
    fig.savefig("score_x_apisodes_smorthed.png")
    plt.show()