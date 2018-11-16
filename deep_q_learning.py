import gym
from deep_q_learning_model import DeepQLearningModel
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

RENDER_ENV = False
EPISODES = 500
rewards = []
RENDER_REWARD_MIN = 800
total_steps_counter = 0

if __name__ == '__main__':
    model = DeepQLearningModel(input_num=4, output_num=2, agents_num=1, memory_size=2000, batch_size=32)

    for episode in range(400):
        observation = env.reset()
        episode_reward = 0

        while True:
            if RENDER_ENV:
                env.render()

            actions = model.get_actions([observation])
            new_observation, reward, done, info = env.step(actions[0])
            x, x_dot, theta, theta_dot = new_observation
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            step_reward = r1 + r2

            model.store_memory([observation], actions, [step_reward], [new_observation])
            episode_reward += step_reward

            if total_steps_counter > 1000:
                model.post_step_actions([observation], actions)

            if done:
                rewards.append(episode_reward)
                max_reward_so_far = np.amax(rewards)
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", round(episode_reward, 2))
                print("Epsilon: ", round(model.epsilon, 2))
                print("Max reward so far: ", max_reward_so_far)

                if max_reward_so_far > RENDER_REWARD_MIN:
                    RENDER_ENV = True
                break
            observation = new_observation
            total_steps_counter += 1
