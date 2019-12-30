

class DensePredatorAgentConfig:
    model_dir = r'/media/dorliv/50D03BD3D03BBE52/Study/Masters/MARL_simple/saved_model/predator'
    best_model = r'/media/dorliv/50D03BD3D03BBE52/Study/Masters/MARL_simple/saved_model/predator_best'
    gamma = 0.95
    epochs = 100000
    buffer_length = int(1e6)
    steps_per_update = 100
    batch_size = 1024
    # n_exploration_eps = 50000
    # init_noise_scale = 1.0
    # final_noise_scale = 0
    save_interval = 1000
    hidden_dim = 64
    tau = 0.01
    episode_length = 35
    environment_solved_objective = 50

    @staticmethod
    def learning_rate_schedule(epoch: int):
        if epoch <= 90000:
            return 1e-2
        if 15000 < epoch <= 30000:
            return 5e-4
        if 30000 < epoch <= 75000:
            return 1e-5
        if 75000 < epoch <= 90000:
            return 5e-6
        else:
            return 1e-3

    @staticmethod
    def temperature(epoch: int):
        if epoch <= 2500:
            return 1000
        if 2500 < epoch <= 5000:
            return 10
        if 5000 < epoch <= 10000:
            return 5.0
        if 10000 < epoch <= 20000:
            return 1.0
        if 20000 < epoch <= 40000:
            return 0.1
        else:
            return 1e-10