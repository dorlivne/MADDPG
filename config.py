

class DensePredatorAgentConfig:
    model_dir = r'/media/dorliv/50D03BD3D03BBE52/Study/Masters/MARL_simple/saved_model/predator'
    ready_model = r'/media/dorliv/50D03BD3D03BBE52/Study/Masters/MARL_simple/saved_model/predator_ready'
    logs_dir = r'/media/dorliv/50D03BD3D03BBE52/Study/Masters/MARL_simple/logs/'
    gamma = 0.95
    epochs = 100000
    buffer_length = int(1e6)
    steps_per_update = 100
    batch_size = 1024
    # n_exploration_eps = 50000
    # init_noise_scale = 1.0
    # final_noise_scale = 0
    save_interval = 1000
    hidden_dim = 128
    tau = 0.01
    episode_length_start = 25
    episode_length_end = 40
    environment_solved_objective = 16
    pruning_start = 0
    pruning_end = -1
    target_sparsity = 0.9
    sparsity_start = 0
    sparsity_end = int(10e5)
    pruning_freq = int(10)
    initial_sparsity = 0

    @staticmethod
    def learning_rate_schedule(epoch: int):
        # if epoch <= 5000:
        #     return 1e-1
        # if 5000 < epoch <= 10000:
        #     return 5e-2
        if epoch <= 30000:
            return 0.01
        if 50000 < epoch <= 60000:
            return 5e-3
        if 60000 < epoch <= 70000:
            return 1e-3
        if 70000 < epoch <= 80000:
            return 5e-4
        if 80000 < epoch <= 90000:
            return 5e-4
        else:
            return 1e-4

