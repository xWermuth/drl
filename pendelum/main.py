import gym
import numpy as np
import tensorflow as tf
from agent import Agent
from data import get_training_data

ENV_NAME = "CartPole-v1"
NUM_ITERATIONS = 100
GOAL_STEPS = 50

# https://medium.com/velotio-perspectives/exploring-openai-gym-a-platform-for-reinforcement-learning-algorithms-380beef446dc#:~:text=According%20to%20the%20OpenAI%20Gym,has%20an%20environment%2Dagent%20arrangement.

def main():
    env = gym.make(ENV_NAME, render_mode="byte_array")
    env.reset()

    # for _ in range(NUM_ITERATIONS):
    #     env.render()
    #     env.step(env.action_space.sample())

    # Train model
    agent = train(env)


    # # Test model
    # for _ in range(10):
    #     for _ in range(GOAL_STEPS):
    #         env.render()
    #         env.step(env.action_space.sample())
    

    env.close()

def train(env: gym.Env):
    scores, training_data = get_training_data(env, NUM_ITERATIONS, GOAL_STEPS)
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = np.array([i[1] for i in training_data])
    agent = Agent(len(X[0]))
    agent.train(X, y)
    return agent

if __name__ == "__main__":
    main()