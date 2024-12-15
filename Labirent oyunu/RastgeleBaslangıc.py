

import numpy as np
import random
import matplotlib.pyplot as plt

class Labirent:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0]
        ])
        self.start = self.random_start()
        self.state = self.start

    def random_start(self):
        # labirentte geçilebilenden bir tanesini seçer ve başlangıç noktası olarak ayarlar
        zero_positions = list(zip(*np.where(self.grid == 0)))
        return random.choice(zero_positions)

    def reset(self):
        self.state = self.random_start()
        return self.state

    def step(self, action):
        # adımları sayısal olarak adlandırırız.
        actions = {
            0: (-1, 0),  # yukarı
            1: (1, 0),   # aşağı
            2: (0, -1),  # sola
            3: (0, 1)    # sağa
        }
        #Ajanın mevcut durumu (self.state) ve seçilen hareketin etkisi (actions[action]) birleştirilerek bir sonraki durum (next_state) hesaplanır.
        next_state = (
            self.state[0] + actions[action][0],
            self.state[1] + actions[action][1]
        )
        # eğer ajan engele gitmemişse yer değiştirir
        if (0 <= next_state[0] < self.grid.shape[0] and
            0 <= next_state[1] < self.grid.shape[1] and
            self.grid[next_state] != 1):
            self.state = next_state
        else:
            next_state = self.state  # Duvara çarparsa yerinde kalır
            # hedefe ulaşırsa +10 ulaşamazsa -1 puan alır.
        if self.grid[next_state] == 2:
            return next_state, 10, True
        else:
            return next_state, -1, False

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros(state_space + (action_space,))
        self.alpha = alpha  
        self.gamma = gamma     #state_space: Durum uzayının boyutları (örneğin, bir 5x5 labirent için (5, 5)).
                             #action_space: Eylem sayısı (örneğin, dört yön: yukarı, aşağı, sağ, sol).
                             #q_table: Durum-eylem çiftleri için ödülleri saklayan bir tablo.
                             #alpha: Öğrenme hızı (yeni bilgilere ne kadar önem verileceğini belirler).
                        #gamma: İndirim oranı (gelecekteki ödüllerin ne kadar değerli olduğunu ifade eder).
                         #epsilon: Keşfetme-sömürme oranını kontrol eder (rastgele keşif yapma olasılığı).
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:  
            return random.randint(0, 3)  # Keşif
        else:
            return np.argmax(self.q_table[state])  # Sömürü

    def update(self, state, action, reward, next_state, done):
        q_predict = self.q_table[state][action]  # mevcut q
        if done:
            q_target = reward    # hedef q
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (q_target - q_predict) #Q-tablosu Güncellemesi: Yeni bilgi, öğrenme hızı ile eski bilgiye eklenir.

    def decay_epsilon(self, episode, total_episodes, min_epsilon=0.01):
        # Epsilon değerini azaltarak ajanın keşfetme oranını zamanla düşürür,zamanla keşfetme oranını azaltır.
        self.epsilon = max(min_epsilon, self.epsilon * (0.995 ** episode))

def render(env, agent_position, reward, total_reward, step_count, epsilon):
    grid = env.grid.copy()   # çizim fonksiyonu
    grid[agent_position] = 3  # Ajanın olduğu yeri göster
    plt.imshow(grid, cmap="cool", interpolation="nearest")
    plt.xticks([])  # X eksenini gizle
    plt.yticks([])  # Y eksenini gizle
    plt.title(f"Adım: {step_count} | Ödül: {reward} | Toplam Ödül: {total_reward} | Epsilon: {epsilon:.2f}")
    plt.pause(0.3)

# Eğitim
env = Labirent()
agent = QLearningAgent(state_space=(5, 5), action_space=4)

episodes = 100  # Eğitim sırasında görselleştirme için daha az bölüm
success_count = 0  # Başarı sayacı

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"\nBölüm {episode + 1} Başlıyor...")
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)

        # Görselleştirme
        total_reward += reward
        step_count += 1
        render(env, state, reward, total_reward, step_count, agent.epsilon)

        print(f"Adım {step_count}: Hareket={action}, Ödül={reward}, Toplam Ödül={total_reward}, Epsilon={agent.epsilon:.2f}")
        state = next_state

    if done and total_reward > 0:  # Hedefe ulaşıldıysa
        success_count += 1

    # Epsilon'u güncelle
    agent.decay_epsilon(episode, episodes)

    print(f"Bölüm {episode + 1} Tamamlandı: Toplam Ödül={total_reward}, Başarı: {'Evet' if total_reward > 0 else 'Hayır'}")

# Eğitim oranını hesapla ve göster
success_rate = (success_count / episodes) * 100
print(f"\nEğitim Tamamlandı. Başarı Oranı: %{success_rate:.2f}")
plt.show()