# import numpy as np
# import tensorflow as tf
# import os
# from keras.models import Model
# from keras.layers import Dense, Input, Add
# from keras.optimizers import RMSprop
# from keras.initializers import glorot_normal
#
#
# class DQN_ALOHA:
#     """针对ALOHA共存的DQN智能体"""
#
#     def __init__(self, state_size, n_actions, n_nodes,
#                  state_action_memory_size, memory_size=500,
#                  replace_target_iter=200, batch_size=32,
#                  learning_rate=0.01, gamma=0.95,
#                  epsilon=0.1, epsilon_min=0.001, epsilon_decay=0.996):
#
#         # 超参数
#         self.state_size = state_size
#         self.n_actions = n_actions
#         self.n_nodes = n_nodes
#         self.state_action_memory_size = state_action_memory_size
#         self.memory_size = memory_size
#         self.replace_target_iter = replace_target_iter
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#
#         # 经验回放缓冲区
#         self.state_action_memory = np.zeros((self.state_action_memory_size, self.state_size + 1))
#         self.memory = np.zeros((self.memory_size, self.state_size * 2 + 2))
#
#         # 计数器
#         self.learn_state_action_counter = 0
#         self.learn_step_counter = 0
#         self.memory_couter = 0
#
#         # 构建神经网络
#         self.model = self.build_DenseNet_model()
#         self.target_model = self.build_DenseNet_model()
#
#     def build_DenseNet_model(self):
#         """构建深度神经网络"""
#         inputs = Input(shape=(self.state_size,))
#
#         # 特征提取层
#         h1 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=247))(inputs)
#         h2 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=2407))(h1)
#
#         # 残差连接层
#         h3 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=2403))(h2)
#         h4 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24457))(h3)
#         add1 = Add()([h4, h2])
#
#         h5 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24657))(add1)
#         h6 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=27567))(h5)
#         add2 = Add()([h6, add1])
#
#         # 输出层
#         outputs = Dense(self.n_actions, kernel_initializer=glorot_normal(seed=242147))(add2)
#         model = Model(inputs=inputs, outputs=outputs)
#         model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
#
#         return model
#
#     def choose_action(self, state):
#         """选择动作（ε-greedy策略）"""
#         state = state[np.newaxis, :]
#
#         # 衰减探索率
#         self.epsilon *= self.epsilon_decay
#         self.epsilon = max(self.epsilon_min, self.epsilon)
#
#         if np.random.random() < self.epsilon:
#             return np.random.randint(0, self.n_actions)  # 探索
#         else:
#             action_values = self.model.predict(state)
#             return np.argmax(action_values[0])  # 利用
#
#     def store_transition(self, s, a, r, s_):
#         """存储转移经验"""
#         if not hasattr(self, 'memory_couter'):
#             self.memory_couter = 0
#
#         transition = np.concatenate((s, [a, r], s_))
#         index = self.memory_couter % self.memory_size
#         self.memory[index, :] = transition
#         self.memory_couter += 1
#
#     def repalce_target_parameters(self):
#         """更新目标网络参数"""
#         weights = self.model.get_weights()
#         self.target_model.set_weights(weights)
#
#     def learn(self):
#         """训练DQN网络"""
#         # 更新目标网络
#         if self.learn_step_counter % self.replace_target_iter == 0:
#             self.repalce_target_parameters()
#         self.learn_step_counter += 1
#
#         # 经验采样
#         if self.memory_couter > self.memory_size:
#             sample_index = np.random.choice(self.memory_size, size=self.batch_size)
#         else:
#             sample_index = np.random.choice(self.memory_couter, size=self.batch_size)
#
#         # 延迟奖励经验重组
#         sample_index1 = []
#         sample_index2 = []
#         for i in sample_index:
#             if i >= self.memory_size - self.state_action_memory_size:
#                 if self.memory_couter > self.memory_size:
#                     idx = np.random.randint(0, self.memory_size - self.state_action_memory_size)
#                     sample_index1.append(idx)
#                     sample_index2.append(idx + self.state_action_memory_size - 1)
#                 else:
#                     idx = np.random.randint(0, self.memory_couter - self.state_action_memory_size)
#                     sample_index1.append(idx)
#                     sample_index2.append(idx + self.state_action_memory_size - 1)
#             else:
#                 sample_index1.append(i)
#                 sample_index2.append(i + self.state_action_memory_size - 1)
#
#         # 提取批次数据
#         batch_memory1 = self.memory[sample_index1, :]
#         batch_memory2 = self.memory[sample_index2, :]
#
#         state = batch_memory1[:, :self.state_size]
#         action = batch_memory1[:, self.state_size].astype(int)
#         reward = batch_memory2[:, self.state_size + 1]
#         next_state = batch_memory2[:, -self.state_size:]
#
#         # 计算Q目标值
#         q_eval = self.model.predict(state)
#         q_next = self.target_model.predict(next_state)
#         q_target = q_eval.copy()
#
#         batch_index = np.arange(self.batch_size, dtype=np.int32)
#         q_target[batch_index, action] = reward + self.gamma * np.max(q_next, axis=1)
#
#         # 训练网络
#         self.model.fit(state, q_target, batch_size=self.batch_size, epochs=1, verbose=0)




# Run_aloha.py (with training + greedy evaluation)
from env_aloha import ENVIRONMENT_ALOHA
from Brain import DQN
import numpy as np
import time
import os


def rollout(env, dqn_agent, steps, training=True,
            learn_after=200, learn_every=1,
            store_after_delay=True):
    """
    Run 'steps' slots.
    - training=True: store transitions + learn
    - training=False: greedy policy, no store, no learn
    Returns: (agent_success_list, aloha_success_list, total_success_list)
    """
    agent_reward_list = []
    aloha_reward_list = []
    total_reward_list = []

    agent_state_list = []
    agent_action_list = []
    counter2 = 0

    state = env.reset()

    for i in range(steps):
        agent_action = dqn_agent.choose_action(state)
        observation_, reward, agent_reward, aloha_reward = env.step(agent_action)

        # build next_state (same pattern as your original Run.py)
        if i < env.DRL_delay * 2 - 1:
            next_state = env.reset()
        else:
            next_state = np.concatenate([
                agent_state_list[counter2][2:],
                [agent_action_list[counter2], observation_]
            ])
            counter2 += 1

        if training:
            # store transition only after delay is "valid"
            if store_after_delay and i >= env.DRL_delay * 2 - 1:
                dqn_agent.store_transition(state, agent_action, reward, next_state)

            # learn
            if i > learn_after and (i % learn_every == 0):
                dqn_agent.learn()

        agent_state_list.append(state)
        agent_action_list.append(agent_action)

        agent_reward_list.append(agent_reward)
        aloha_reward_list.append(aloha_reward)
        total_reward_list.append(reward)

        state = next_state

    return agent_reward_list, aloha_reward_list, total_reward_list


def main(train_steps=50000, eval_steps=20000, reward_dir="rewards"):
    start = time.time()
    os.makedirs(reward_dir, exist_ok=True)

    # ----------------
    # 1) TRAIN PHASE
    # ----------------
    agent_tr, aloha_tr, total_tr = rollout(
        env, dqn_agent,
        steps=train_steps,
        training=True,
        learn_after=200,
        learn_every=1
    )

    # save training traces
    with open(os.path.join(reward_dir, "agent_train.txt"), "w") as f:
        f.write(" ".join(map(str, agent_tr)))
    with open(os.path.join(reward_dir, "aloha_train.txt"), "w") as f:
        f.write(" ".join(map(str, aloha_tr)))

    tail = 2000 if len(agent_tr) >= 2000 else len(agent_tr)
    print('-----------------------------')
    print('[TRAIN] average agent reward:', float(np.mean(agent_tr[-tail:])))
    print('[TRAIN] average aloha reward:', float(np.mean(aloha_tr[-tail:])))
    print('[TRAIN] average total reward:', float(np.mean(total_tr[-tail:])))
    print('-----------------------------')

    # ------------------------
    # 2) EVAL PHASE (GREEDY)
    # ------------------------
    # IMPORTANT: Brain.py choose_action() always:
    #   epsilon *= epsilon_decay
    #   epsilon = max(epsilon_min, epsilon)
    # so we must force epsilon_min=0 and epsilon_decay=1 to keep epsilon=0.
    dqn_agent.epsilon = 0.0
    dqn_agent.epsilon_min = 0.0
    dqn_agent.epsilon_decay = 1.0

    agent_ev, aloha_ev, total_ev = rollout(
        env, dqn_agent,
        steps=eval_steps,
        training=False
    )

    # save eval traces
    with open(os.path.join(reward_dir, "agent_eval.txt"), "w") as f:
        f.write(" ".join(map(str, agent_ev)))
    with open(os.path.join(reward_dir, "aloha_eval.txt"), "w") as f:
        f.write(" ".join(map(str, aloha_ev)))

    tail2 = 2000 if len(agent_ev) >= 2000 else len(agent_ev)
    print('[EVAL]  average agent reward:', float(np.mean(agent_ev[-tail2:])))
    print('[EVAL]  average aloha reward:', float(np.mean(aloha_ev[-tail2:])))
    print('[EVAL]  average total reward:', float(np.mean(total_ev[-tail2:])))
    print('Time elapsed:', time.time() - start)


if __name__ == "__main__":
    # 这里改 p：aloha_tx_prob 就是 ALOHA 的发送概率
    env = ENVIRONMENT_ALOHA(
        state_size=60,
        n_aloha_nodes=1,
        aloha_tx_prob=0.6,   # <<< p 在这里改
        seed=42,
        history_len=200
    )

    dqn_agent = DQN(
        env.state_size,
        env.n_actions,
        env.n_nodes,
        state_action_memory_size=2 * (env.DRL_delay),
        memory_size=500,
        replace_target_iter=200,
        batch_size=32,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.001,
        epsilon_decay=0.996,
    )

    main(train_steps=50000, eval_steps=20000, reward_dir="rewards")
