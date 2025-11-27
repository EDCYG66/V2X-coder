import numpy as np

class ReplayMemory:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.memory_size = 500000
        # 为节省显存/内存，底层可用 float16 存，但采样时转为 float32
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.prestate = np.empty((self.memory_size, 102), dtype=np.float16)
        self.poststate = np.empty((self.memory_size, 102), dtype=np.float16)
        self.batch_size = 2000
        self.count = 0
        self.current = 0

    def size(self):
        return self.count

    def add(self, prestate, poststate, reward, action):
        self.actions[self.current] = int(action)
        self.rewards[self.current] = np.float32(reward)
        self.prestate[self.current] = np.asarray(prestate, dtype=np.float16)
        self.poststate[self.current] = np.asarray(poststate, dtype=np.float16)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        if self.count == 0:
            raise RuntimeError("ReplayMemory is empty.")
        idxs = np.random.randint(self.count, size=self.batch_size)
        prestate = self.prestate[idxs].astype(np.float32, copy=False)   # (B, 102)
        poststate = self.poststate[idxs].astype(np.float32, copy=False) # (B, 102)
        actions = self.actions[idxs].astype(np.int32, copy=False)       # (B,)
        rewards = self.rewards[idxs].astype(np.float32, copy=False)     # (B,)
        return prestate, poststate, actions, rewards