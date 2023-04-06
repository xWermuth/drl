from gym import Env
import numpy as np
import random


SCORE_GOAL = 50

def get_training_data(env: Env, num_games: int, goal_steps: int):
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(num_games):
        score = 0
        episodes = []
        prev_observation = None
        for _ in range(goal_steps):
            # Rand action between left or right
            action = random.randrange(0, 2)
            # env.render()
            observation, reward, truncated, done, info = env.step(action)

            if prev_observation is not None:
                episodes.append([prev_observation, action])
            prev_observation = observation
            score += reward

            if done:
                break

        if score >= SCORE_GOAL:
            # print(f"Score satiesfied {score}")
            accepted_scores.append(score)

            for data in episodes:
                hot_ones = []
                # Action left is 1 and action right is 2
                if data[1] == 1:
                    hot_ones = [0, 1]
                elif data[1] == 0:
                    hot_ones = [1, 0]
                else:
                    print(f"UNEXPECTED ACTION: {data[1]}")
                
                training_data.append([data[0], hot_ones])

        env.reset()
        scores.append(score)

    return scores, training_data
                
