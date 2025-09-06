# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with Grid World with Wind.
## PROBLEM STATEMENT
The agent is placed in a 5x5 grid where it must navigate to a goal position. However, in certain columns, wind affects the agent’s movement, pushing it upward. The challenge is to find an optimal policy that maximizes the cumulative reward while considering the stochastic effects of wind.

## POLICY EVALUATION FUNCTION:
<img width="1012" height="172" alt="image" src="https://github.com/user-attachments/assets/ffe687af-7bd1-4af8-94c8-1a18ee00c128" />


## PROGRAM:
```
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

init_state

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))

pi_2 = lambda s: {
    0:LEFT,
    1:RIGHT,
    2:DOWN,
    3:UP,
    4:LEFT,
    5:DOWN,
    6:RIGHT,
    7:UP,
    8:RIGHT,
    9:DOWN,
    10:UP,
    11:DOWN,
    12:RIGHT,
    13:UP,
    14:UP,
    15:RIGHT #stop
}[s]
print("Name:Daniel c")
print("Register Number:212223240023")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))

print("Policy 1 (FrozenLake mapping):")
print("  Success Rate: {:.2f}%".format(probability_success(env, pi_frozenlake, goal_state=goal_state) * 100))
print("  Avg Return: {:.4f}".format(mean_return(env, pi_frozenlake)))

print("\nPolicy 2 :")
print("  Success Rate: {:.2f}%".format(probability_success(env, pi_2, goal_state=goal_state) * 100))
print("  Avg Return: {:.4f}".format(mean_return(env, pi_2)))

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

V2 = policy_evaluation(pi_2, P,gamma=0.99)
print_state_value_function(V2, P, n_cols=4, prec=5)

V1>=V2

if(np.sum(V1>=V2)>=11):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)>=11):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")[ex_02_rl.py](https://github.com/user-attachments/files/22184944/ex_02_rl.py)
```

## OUTPUT:
<img width="465" height="155" alt="image" src="https://github.com/user-attachments/assets/c0303d9f-15fd-4833-950e-212153ef1dcb" />

<img width="602" height="142" alt="image" src="https://github.com/user-attachments/assets/dac0f183-afb9-47aa-9029-2c90e407dcf6" />

<img width="492" height="115" alt="image" src="https://github.com/user-attachments/assets/71d0f164-720c-4f20-87c5-04e67c3093c4" />

<img width="523" height="131" alt="image" src="https://github.com/user-attachments/assets/f4a33e7e-17a2-44a9-8204-8751ef92bc67" />

<img width="646" height="78" alt="image" src="https://github.com/user-attachments/assets/7acff69a-56d4-43eb-8eec-8b99ab4ed006" />

<img width="362" height="51" alt="image" src="https://github.com/user-attachments/assets/a1b74061-3b93-49af-8adb-ba3ef262dc59" />


## RESULT:
Thus, the Given Policy has been Evaluated and Optimal Policy has been Computed using Python Programming and execcuted successfully.
