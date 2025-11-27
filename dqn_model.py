import tensorflow as tf

class MinClipSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_schedule, min_lr=1e-6):
        super().__init__()
        self.base_schedule = base_schedule
        self.min_lr = tf.cast(min_lr, tf.float32)
    def __call__(self, step):
        return tf.maximum(self.base_schedule(step), self.min_lr)

class DQNModel:
    def __init__(self, input_dim=102, output_dim=60,
                 learning_rate=0.01, decay_steps=500000,
                 decay_rate=0.96, min_lr=0.0005,
                 grad_clip_norm=5.0):
        self.learning_rate = learning_rate
        self.learning_rate_decay = decay_rate
        self.learning_rate_decay_step = decay_steps
        self.learning_rate_minimum = min_lr
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grad_clip_norm = grad_clip_norm
        self.initialize_network()
        self.compile_model()
        self.tb_writer = None

    def initialize_network(self):
        self.model = self.build_dqn()
        self.target_model = self.build_dqn()
        _ = self.model(tf.zeros((1, self.input_dim)), training=False)

    def build_dqn(self):
        n_input = self.input_dim
        n_output = self.output_dim
        he = tf.keras.initializers.HeNormal()
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_input,)),
            tf.keras.layers.Dense(700, activation='relu', kernel_initializer=he),
            tf.keras.layers.Dense(350, activation='relu', kernel_initializer=he),
            tf.keras.layers.Dense(180, activation='relu', kernel_initializer=he),
            tf.keras.layers.Dense(n_output, activation=None,  # 线性输出
                                  kernel_initializer=tf.keras.initializers.RandomUniform(-0.01, 0.01))
        ])
        return model

    def forward(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.input_dim])
        return self.model(inputs, training=True)

    def forward_target(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.input_dim])
        return self.target_model(inputs, training=False)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def compile_model(self):
        base_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.learning_rate_decay_step,
            decay_rate=self.learning_rate_decay,
            staircase=True
        )
        lr_schedule = MinClipSchedule(base_lr_schedule, min_lr=self.learning_rate_minimum)
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=lr_schedule, rho=0.95, epsilon=0.01
        )
        # 使用 Huber 稳定 loss
        self.loss_fn = tf.keras.losses.Huber(delta=1.0)

    def current_lr(self):
        lr = self.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return float(lr(self.optimizer.iterations).numpy())
        return float(tf.convert_to_tensor(lr).numpy())

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ],
    )
    def train_step(self, inputs, targets, actions):
        with tf.GradientTape() as tape:
            q_values = self.model(inputs, training=True)
            num_actions = tf.shape(q_values)[1]
            action_masks = tf.one_hot(actions, num_actions, dtype=q_values.dtype)
            q_acted = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = self.loss_fn(targets, q_acted)
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, q_values