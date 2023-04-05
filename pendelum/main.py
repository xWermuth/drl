import gym
import tensorflow as tf

ENV_NAME = "CartPole-v1"
NUM_ITERATIONS = 250
COLLECT_EPISODES_PER_ITERATIONS = 2
REPLAY_BUFFER_CAPACITY = 2000
FC_LAYER_PARAMS = (100,)
LOG_INTERVAL = 25
NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 50

def main():
    env = gym.make(ENV_NAME, render_mode="human")
    env.reset()

    for _ in range(NUM_ITERATIONS):
        env.render()
        env.step(env.action_space.sample())

    env.close()


if __name__ == "__main__":
    main()