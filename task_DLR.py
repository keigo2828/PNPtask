import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


class World(object):
    def __init__(self,LR,RR):
        self.map = np.array([0,1,2,3])#0LN1,1LL,2LN2,3LR
        self.actions = np.array([1,2,3])#1HL,2HR,3LN遷移  
        self.start_position = np.array([0])
        self.agent_position = self.start_position
        self.LR=LR#lefrt rewards
        self.RR=RR#right rewards

    def reset(self):
        self.agent_position = self.start_position
        return self.agent_position

    def update_state(self, action):
        if self.agent_position != [0]:#LNに遷移
            self.agent_position = np.zeros(1)
            return np.zeros(1)
        if action == 0 :#LL,LNに0.5で遷移
            x = np.random.binomial(1,0.5)
            if x == 0:
              self.agent_position += self.actions[x]
              return self.agent_position
            elif x == 1:
                self.agent_position += self.actions[x]
                return self.agent_position
        elif action == 1:#LRに遷移
            self.agent_position += self.actions[2]
            return self.agent_position

    def step(self, action):
        reward = self.reward()
        next_state = self.update_state(action)
        return reward , next_state

    def reward(self):
        if self.agent_position == 1:#LL
            return self.LR
        
        elif self.agent_position== 2:#LN2
            return 0
        elif self.agent_position == 3:#LR
            x = np.random.binomial(1,0.5)
            if x == 0:
                return 0
            elif x == 1:
                return self.RR
        else:#LN1
            return 0

class Random(object):
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def act(self):
        return random.randint(0, 1)
    
    def update(self, reward, next_state):
        pass



class QLearning(object):
    def __init__(self):
        self.alpha = 0.8
        self.alpha_p = 0.8
        self.alpha_m = 0.8/6.7
        self.k_p = 3
        self.k_m = 0
        self.forget = 0

        self.gamma = 0.9
        self.eps = 0.1
        self.actions = np.array([0,1,2])#LL,LR,LN1に遷移
        self.Q = np.zeros(5)#Q値(LN1+HL,LN1+HR,LL+S,LN2+S,LR+S)
        self.reliability = np.zeros(5)#主観報酬確率
        self.enter_ct = np.zeros(5)
        self.ct = np.zeros(5)
        self.last_softmax_p =np.zeros(2)

    def reset(self):
        self.Q = np.zeros(5)
        self.reliability = np.zeros(5)#主観報酬確率
        self.enter_ct = np.zeros(5)
        self.ct = np.zeros(5)

    def act(self, state):
        self.state = state
        ### e_greedy or softmax
        # self.action = self.e_greedy()
        self.action = self.softmax()
        return self.action

    def update(self, action, state, next_state, reward):
        if action == 2:#状態がLNに遷移する時の行動LL,LN,LRの時に起きる
            self.enter_ct[int(state)+1] +=1
            alpha = self.alpha_m
            if reward > 0:
                self.ct[int(state)+1] +=1
                self.reliability[int(state)+1] = self.ct[int(state)+1]/self.enter_ct[int(state)+1]
                alpha = self.alpha_p
            max_Q = max(self.Q[:2])
            # sub_reward = reward * (self.reliability[int(state)+1]**2) #主観reward
            # td_error = sub_reward + self.gamma * max_Q  - self.Q[int(state)+1]
            td_error = reward + self.gamma * max_Q  - self.Q[int(state)+1]

            self.Q[int(state) + 1] += alpha * td_error
        else:#状態がLNの時にLL,LN,LRに遷移する時
            self.enter_ct[action] +=1
            alpha = self.alpha_m
            if reward > 0:
                self.ct[action] +=1
                self.reliability[action] = self.ct[action]/self.enter_ct[action]

            max_Q = self.Q[int(next_state) + 1]
            # sub_reward = reward * (self.reliability[action]**2)#主観reward
            # td_error = sub_reward + self.gamma * max_Q  - self.Q[action]
            td_error = reward + self.gamma * max_Q  - self.Q[action]
            self.Q[action] += alpha * td_error


 
    def e_greedy(self):
        if random.random() < self.eps:
            action = random.randint(0, 3)
        else: 
            action = self.greedy()
        return action

    def greedy(self):
        if self.state == [0]:
            CanChoiceQ = np.array(self.Q[:2])
            maxIndex = np.argmax(CanChoiceQ)
            return random.choice(maxIndex)
        else:
            return 2

    def softmax(self):
        if self.state ==[0]:
            CanChoiceQ = np.array(self.Q[:2])
            x = np.exp(CanChoiceQ)
            u = np.sum(x)
            p_softmax = x/u
            #print(p_softmax)
            self.last_softmax_p = p_softmax
            return np.random.choice([0, 1], p=p_softmax)
        else:
            return 2


A = World(1,1)
B =Random()
Q = QLearning()
sim = 100
Q_list =[]
P_list = []

for j in range(sim):
    A.reset()
    Q.reset()
    for i in range(100):
        act = Q.act(A.agent_position)
        state=A.agent_position 
        reward , next_state =  A.step(act)
        Q.update(act,state, next_state, reward)
    Q_list.append(Q.Q)
    P_list.append(Q.last_softmax_p)

Q_list = np.array(Q_list)
Q_list = Q_list.mean(axis=0)

P_list = np.array(P_list)
P_list = P_list.mean(axis=0)

print(Q_list)
print(P_list)
# print(Q.ct)
# print(Q.enter_ct)
# print(Q.reliability)
print(Q.last_softmax_p)

# df= pd.Series(Q_list, index=['Q_HL','Q_HR','LL_s','LN2_s','LR_s'])
# df1= pd.Series(P_list, index=['Q_HL','Q_HR','LL_s','LN2_s','LR_s'])
# df = pd.DataFrame(df)
# df = df.transpose()
# print(df)

# # print(df.transpose())
# df.to_csv('bias_Q_list_mean_sim100_2000.csv')



