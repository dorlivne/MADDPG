from utilss.tensorflow_utils import _build_fc_layer, reshape_input_for_critic, transform_one_hot_batch, transform_one_hot
import tensorflow as tf
import numpy as np
from tensorflow.contrib.model_pruning.python import pruning
import os
from config import DensePredatorAgentConfig as cfg
from gym.spaces import Box

from utilss.gumbel import gumbel_softmax



class BaseNetwork:
    """
    abstract class for a Neural Network
    """
    def __init__(self, model_path):
        self.graph = tf.Graph()
        self.model_path = model_path

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            # to save GPU resources
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config, graph=self.graph)
        return self._sess

    def initialize(self):
        with self.graph.as_default():
            self.init_variables(tf.global_variables())

    def init_variables(self, var_list):
        self.sess.run(tf.variables_initializer(var_list))

    def number_of_parameters(self, var_list):
        return sum(np.prod(v.get_shape().as_list()) for v in var_list)

    def print_num_of_params(self):
        with self.graph.as_default():
            print('Number of parameters (four bytes == 1 parameter): {}.\n'.format(
                int(self.number_of_parameters(tf.trainable_variables()))))
            return int(self.number_of_parameters(tf.trainable_variables()))

    # need to create a saver after the graph has been established to use these functions
    def save_model(self, saver, path=None, sess=None, global_step=None):
        save_dir = path or self.model_path
        os.makedirs(save_dir, exist_ok=True)
        saver.save(sess or self.sess,
                        os.path.join(save_dir, 'model.ckpt'),
                        global_step=global_step)
        return self

    def load_model(self, saver, path=None, sess=None):
        path = path or self.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt is None:
            raise FileNotFoundError('Can`t load a model. Checkpoint does not exist.')
        restore_path = ckpt.model_checkpoint_path
        saver.restore(sess or self.sess, restore_path)

        return self

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, trainable=True)
        return var

    def _variable_with_weight_decay(self, name, shape, wd, initialization):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        var = self._variable_on_cpu(
            name,
            shape,
            initialization)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var





class MADDPG(BaseNetwork):
    def __init__(self, actor_input_dim, actor_output_dim, critic_state_dim, critic_action_dim,
                 critic_other_action_dim, model_path,  tau=cfg.tau, maddpg=True):
        super(MADDPG, self).__init__(model_path)
        self.actor_input_dim = (None, actor_input_dim)
        self.actor_output_dim = (None, actor_output_dim)
        self.critic_state_dim = (None, critic_state_dim)
        self.critic_action_dim = (None, critic_action_dim)
        self.maddpg = maddpg
        self.critic_input_dim = critic_state_dim + critic_action_dim
        if maddpg:
            self.critic_other_action_dim = (None, critic_other_action_dim)
            self.critic_input_dim += critic_other_action_dim
        else:
            self.critic_other_action_dim = None
        self.tau = tau
        with self.graph.as_default():
                self.critic_global_step = tf.Variable(0, trainable=False)
                self.actor_global_step = tf.Variable(0, trainable=False)
                self._build_placeholders()
                self.actor_logits = self._build_actor()
                self.gumbel_dist = self._build_gumbel(self.actor_logits)
                self.critic_logits = self._build_critic(self.critic_action)
                self.actor_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
                self.critic_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
                self.actor_pruned_weight_matrices = pruning.get_masked_weights()
                self.critic_loss = self._build_critic_loss()
                self.critic_train_op = self._build_critic_train_op()
                self.actor_train_op = self._build_actor_train_op()
                self.actor_saver = tf.train.Saver(var_list=self.actor_parameters, max_to_keep=100)
                self.critic_saver = tf.train.Saver(var_list=self.critic_parameters, max_to_keep=100)
                self.init_variables(tf.global_variables())
                self.actor_soft_update, self.actor_target_placeholders = self._build_soft_sync_op(self.actor_parameters)
                self.critic_soft_update, self.critic_target_placeholders = self._build_soft_sync_op(self.critic_parameters)
                self.sparsity = pruning.get_weight_sparsity()
                self.hparams = pruning.get_pruning_hparams() \
                    .parse('name={}, begin_pruning_step={}, end_pruning_step={}, target_sparsity={},'
                           ' sparsity_function_begin_step={},sparsity_function_end_step={},'
                           'pruning_frequency={},initial_sparsity={},'
                           ' sparsity_function_exponent={}'.format('Actor',
                                                                   cfg.pruning_start,
                                                                   cfg.pruning_end,
                                                                   cfg.target_sparsity,
                                                                   cfg.sparsity_start,
                                                                   cfg.sparsity_end,
                                                                   cfg.pruning_freq,
                                                                   cfg.initial_sparsity,
                                                                   3))
                # note that the global step plays an important part in the pruning mechanism,
                # the higher the global step the closer the sparsity is to sparsity end
                self.pruning_obj = pruning.Pruning(self.hparams, global_step=self.actor_global_step)
                self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
                # the pruning objects defines the pruning mechanism, via the mask_update_op the model gets pruned
                # the pruning takes place at each training epoch and it objective to achieve the sparsity end HP
                self.init_variables(tf.global_variables())  # initialize variables in graph

    def get_flat_weights(self):
        with self.graph.as_default():
            weights_matrices = self.sess.run(self.actor_pruned_weight_matrices)
            flatten_matrices = []
            for matrix in weights_matrices:
                flatten_matrices.append(np.ndarray.flatten(matrix))
            return flatten_matrices

    def get_number_of_nnz_params(self):
        flatten_matrices = self.get_flat_weights()
        weights = []
        for w in flatten_matrices:
            weights.extend(list(w.ravel()))
        weights_array = [w for w in weights if w != 0]
        return len(weights_array)

    def get_number_of_nnz_params_per_layer(self):
        flatten_matrices = self.get_flat_weights()
        nnz_at_each_layer = []
        for matrix in flatten_matrices:
            nnz_at_each_layer.append(len([w for w in matrix.ravel() if w != 0]))
        return nnz_at_each_layer

    def get_number_of_params(self):
        flatten_matrices = self.get_flat_weights()
        weights = []
        for w in flatten_matrices:
            weights.extend(list(w.ravel()))
        weights_array = [w for w in weights]
        return len(weights_array)

    def prune(self):
        self.sess.run([self.mask_update_op])

    def get_model_sparsity(self):
        sparsity = self.sess.run(self.sparsity)
        return np.mean(sparsity)

    def get_weights(self):
        with self.graph.as_default():
            return self.sess.run([self.actor_parameters, self.critic_parameters])

    def is_maddpg(self):
        return self.maddpg

    def get_critic_value(self, state, action, other_action=None):
        if self.maddpg:
            return self.sess.run(self.critic_logits, feed_dict={self.critic_state: state,
                                                                self.critic_other_action: other_action,
                                                                self.critic_action: action})
        else:
            return self.sess.run(self.critic_logits, feed_dict={self.critic_state: state,
                                                                self.critic_action: action})

    def act(self, state):
        """
        deciding an action via the logits, no exploration(?)
        :param state: actor state
        :return: action in one_hot format
        """
        # actor_logits = self.sess.run(self.actor_logits, feed_dict={self.actor_state: state})
        actor_logits = self.get_q(state)
        return self.get_action_one_hot_from_logits(actor_logits)
        # action = np.argmax(actor_logits, axis=-1)
        # batch_size = np.shape(action)[0]
        # if batch_size == 1:
        #     return transform_one_hot(action, self.actor_output_dim[-1])
        # else:
        #     return transform_one_hot_batch(action, self.actor_output_dim[-1])

    def get_action_one_hot_from_logits(self, actor_logits):
        action = np.argmax(actor_logits, axis=-1)
        batch_size = np.shape(action)[0]
        if batch_size == 1:
            return transform_one_hot(action, self.actor_output_dim[-1])
        else:
            return transform_one_hot_batch(action, self.actor_output_dim[-1])

    def get_q(self, state):
        return self.sess.run(self.actor_logits, feed_dict={self.actor_state: state})

    def get_gumbel(self, state):
        """
        get an action in one_hot format according to a gumbel distribution
        :param state: state
        :return: action distribution
        """
        gumbel_output = self.sess.run(self.gumbel_dist, feed_dict={self.actor_state: state})
        return gumbel_output

    def train_critic(self, state, action, target, learning_rate, other_action=None):
        if self.maddpg:
            _, critic_loss = self.sess.run([self.critic_train_op, self.critic_loss],
                                           feed_dict={self.critic_state: state, self.critic_other_action: other_action,
                                                      self.critic_action: action, self.critic_target: target,
                                                      self.critic_learning_rate: learning_rate})
        else:
            _, critic_loss = self.sess.run([self.critic_train_op, self.critic_loss],
                                           feed_dict={self.critic_state: state,
                                                      self.critic_action: action, self.critic_target: target,
                                                      self.critic_learning_rate: learning_rate})
        return critic_loss

    def train_actor(self, actor_state, critic_state, other_action, learning_rate):
        # graph needs state for actor, state for critic, temperature  and actor learning rate
        if self.maddpg:
            self.sess.run(self.actor_train_op, feed_dict={self.actor_state: actor_state, self.critic_state: critic_state,
                                                          self.critic_other_action: other_action,
                                                          self.actor_learning_rate: learning_rate})
        else:
            self.sess.run(self.actor_train_op,
                          feed_dict={self.actor_state: actor_state, self.critic_state: critic_state,
                                     self.actor_learning_rate: learning_rate})

    def soft_update(self, params, actor_flag):
        with self.graph.as_default():
            if actor_flag:
                target_network_parameters = self.actor_target_placeholders
                update_op = self.actor_soft_update
            else:
                target_network_parameters = self.critic_target_placeholders
                update_op = self.critic_soft_update
            params_dict = {}
            for place_holder, param in zip(target_network_parameters, params):
                params_dict[place_holder] = param
            self.sess.run(update_op, feed_dict=params_dict)

    def save(self, epoch, path=None):
        save_dir = path or self.model_path
        actor_save_dir = save_dir + "/actor_{}".format(epoch)
        self.save_model(saver=self.actor_saver, path=actor_save_dir)
        critic_save_dir = save_dir + "/critic_{}".format(epoch)
        self.save_model(saver=self.critic_saver, path=critic_save_dir)

    def load(self, epoch, path=None):
        save_dir = path or self.model_path
        actor_save_dir = save_dir + "/actor_{}".format(epoch)
        self.load_model(saver=self.actor_saver, path=actor_save_dir)
        critic_save_dir = save_dir + "/critic_{}".format(epoch)
        self.load_model(saver=self.critic_saver, path=critic_save_dir)

    def _build_placeholders(self):
        self.actor_state = tf.placeholder(dtype=tf.float32, shape=self.actor_input_dim, name='actor_state')
        self.critic_target = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='critic_target')
        self.actor_learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='actor_learning_rate')
        self.critic_learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='critic_learning_rate')
        self.critic_state = tf.placeholder(dtype=tf.float32, shape=self.critic_state_dim, name='critic_state')
        if self.maddpg:
            self.critic_other_action = tf.placeholder(dtype=tf.float32, shape=self.critic_other_action_dim, name='critic_other_action')
        self.critic_action = tf.placeholder(dtype=tf.float32, shape=self.critic_action_dim,
                                                  name='critic_action')

    def _build_actor(self, scope='actor'):
        with tf.variable_scope(scope):
            input = self.actor_state
            # input = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(self.actor_state)
            with tf.variable_scope("fc_1") as scope_tf:
                fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.actor_input_dim[-1], cfg.hidden_dim), activation=tf.nn.relu)
            with tf.variable_scope("fc_2") as scope_tf:
                fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(cfg.hidden_dim, cfg.hidden_dim), activation=tf.nn.relu)

            with tf.variable_scope("fc_3") as scope_tf:
                 fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
                                       weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                       shape=(cfg.hidden_dim, self.actor_output_dim[-1]))
            return fc_3

    def _build_critic(self, critic_action, scope='critic', reuse=False):
        with tf.variable_scope(scope) as scope_critic:
            if reuse:
                scope_critic.reuse_variables()
            if self.maddpg:
                critic_input = tf.concat([self.critic_state, critic_action, self.critic_other_action], axis=-1)
            else:
                critic_input = tf.concat([self.critic_state, critic_action], axis=-1)
            input = critic_input
            # input = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(critic_input)
            with tf.variable_scope("fc_1") as scope_tf:
                fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.critic_input_dim, cfg.hidden_dim), activation=tf.nn.relu, apply_prune=False)
            with tf.variable_scope("fc_2") as scope_tf:
                fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(cfg.hidden_dim, cfg.hidden_dim), activation=tf.nn.relu, apply_prune=False)

            with tf.variable_scope("fc_3") as scope_tf:
                fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
                                            weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                            shape=(cfg.hidden_dim, 1), apply_prune=False)
            return fc_3


    def _build_critic_loss(self):
        """
        build loss function = MSE(target, logits)
        :return: a loss fn
        """
        mse = tf.losses.mean_squared_error(labels=self.critic_target,
                                           predictions=self.critic_logits)
        mse = tf.reduce_mean(mse)
        return mse

    def _build_critic_train_op(self):
            """
             build Optimizer in graph
            :return: an optimizer
            """
            # return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
            #                                                                          global_step=self.global_step)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_learning_rate)
            gradients = optimizer.compute_gradients(loss=self.critic_loss, var_list=self.critic_parameters)
            capped_gvs = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in gradients]
            return optimizer.apply_gradients(capped_gvs)

    def _build_actor_train_op(self):
        # graph needs state for actor, state for critic, temperature and actor learning rate
        critic_output = self._build_critic(critic_action=self.gumbel_dist, reuse=True)
        actor_loss = -tf.reduce_mean(critic_output)
        actor_loss += tf.reduce_mean(self.actor_logits ** 2) * 1e-3
        optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_learning_rate)
        gradients = optimizer.compute_gradients(loss=actor_loss, var_list=self.actor_parameters)
        capped_gvs = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in gradients]
        return optimizer.apply_gradients(capped_gvs, global_step=self.actor_global_step)

    def _build_gumbel(self, actor_logits):
        """
        take argmax, but differentiate w.r.t. soft sample y
        :param actor_logits: logits to build a gumbel distribution
        :return: one_hot action
        """
        return gumbel_softmax(logits=actor_logits, hard=True)

    def _build_soft_sync_op(self, network_params):
                params = self.sess.run(network_params)  # DNN weights

                target_network_parameters = [tf.placeholder(dtype=tf.float32, shape=np.shape(network_parameter))
                                                  for network_parameter in params]  # new weights
                # new weights are weighted with tau
                return [tf.assign(parameter, (1 - self.tau) * parameter + self.tau * weight)
                                    for parameter, weight in zip(network_params,  target_network_parameters)], target_network_parameters


class StudentActor(BaseNetwork):
    def __init__(self, actor_input_dim,
                 actor_output_dim,
                 model_path,
                 redundancy=None,
                 last_measure=10e4, tau=0.01):
        super(StudentActor, self).__init__(model_path=model_path)
        self.actor_input_dim = (None, actor_input_dim)
        self.actor_output_dim = (None, actor_output_dim)
        self.tau = tau
        self.redundancy = redundancy
        self.last_measure = last_measure
        with self.graph.as_default():
            self.actor_global_step = tf.Variable(0, trainable=False)
            self._build_placeholders()
            self.actor_logits = self._build_actor()
            # self.gumbel_dist = self._build_gumbel(self.actor_logits)
            self.loss = self._build_loss()
            self.actor_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.actor_pruned_weight_matrices = pruning.get_masked_weights()
            self.actor_train_op = self._build_actor_train_op()
            self.actor_saver = tf.train.Saver(var_list=self.actor_parameters, max_to_keep=100)
            self.init_variables(tf.global_variables())
            self.sparsity = pruning.get_weight_sparsity()
            self.hparams = pruning.get_pruning_hparams() \
                .parse('name={}, begin_pruning_step={}, end_pruning_step={}, target_sparsity={},'
                       ' sparsity_function_begin_step={},sparsity_function_end_step={},'
                       'pruning_frequency={},initial_sparsity={},'
                       ' sparsity_function_exponent={}'.format('Actor',
                                                               cfg.pruning_start,
                                                               cfg.pruning_end,
                                                               cfg.target_sparsity,
                                                               cfg.sparsity_start,
                                                               cfg.sparsity_end,
                                                               cfg.pruning_freq,
                                                               cfg.initial_sparsity,
                                                               3))
            # note that the global step plays an important part in the pruning mechanism,
            # the higher the global step the closer the sparsity is to sparsity end
            self.pruning_obj = pruning.Pruning(self.hparams, global_step=self.actor_global_step)
            self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
            # the pruning objects defines the pruning mechanism, via the mask_update_op the model gets pruned
            # the pruning takes place at each training epoch and it objective to achieve the sparsity end HP
            self.init_variables(tf.global_variables())  # initialize variables in graph

    def save(self, path=None):
        save_dir = path or self.model_path
        self.save_model(saver=self.actor_saver, path=save_dir)

    def load(self, path=None):
        save_dir = path or self.model_path
        self.load_model(saver=self.actor_saver, path=save_dir)

    def act(self, state):
        """
        deciding an action via the logits, no exploration(?)
        :param state: actor state
        :return: action in one_hot format
        """
        # actor_logits = self.sess.run(self.actor_logits, feed_dict={self.actor_state: state})
        actor_logits = self.get_q(state)
        action = np.argmax(actor_logits, axis=-1)
        batch_size = np.shape(action)[0]
        if batch_size == 1:
            return transform_one_hot(action, self.actor_output_dim[-1])
        else:
            return transform_one_hot_batch(action, self.actor_output_dim[-1])

    def get_q(self, state):
        return self.sess.run(self.actor_logits, feed_dict={self.actor_state: state})

    def _build_actor(self):

        raise NotImplementedError

    def _build_placeholders(self):
        self.actor_state = tf.placeholder(dtype=tf.float32, shape=self.actor_input_dim, name='actor_state')
        self.actor_learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='actor_learning_rate')
        self.target_logits = tf.placeholder(dtype=tf.float32, shape=self.actor_output_dim, name='target_dist')

    def _build_actor_train_op(self):
        """
              the suggested optimizer according to Policy distillation
        """
        return tf.train.RMSPropOptimizer(learning_rate=self.actor_learning_rate, name='optimizer').minimize(self.loss)

    def _build_loss(self):
        """
        KLL  policy distillation loss function we want to minimize, according to Policy distillation
        """
        eps = 0.00001
        teacher_sharpend_dist = tf.nn.softmax(self.target_logits / self.tau, dim=1) + eps
        teacher_sharpend_dist = tf.squeeze(teacher_sharpend_dist)
        student_dist = tf.nn.softmax(self.actor_logits, dim=1) + eps
        return tf.reduce_sum(teacher_sharpend_dist * tf.log(teacher_sharpend_dist / student_dist))


class MaddpgWrapper:
    """
    Wrapper class for Multi agent DDPG (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, actor_input_dim, type_of_agents, actor_output_dim, critic_state_dim, critic_action_dim, model_dir):
        """
        Initialize an instance of the MADDPG model
        :param actor_input_dim:
        :param actor_output_dim:
        :param critic_state_dim:
        :param critic_action_dim:
        :param model_dir:
        """

        self.num_of_agents = len(type_of_agents)
        self.agents, self.target_agents = [], []
        self.type_of_agents = type_of_agents
        for i, type in enumerate(type_of_agents):
            if type == "adversary":  # MADDPG
                critic_other_action_dim = critic_action_dim - actor_output_dim[i]
                self.agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=critic_state_dim, critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=critic_other_action_dim, maddpg=True,
                                          model_path=model_dir + "_adv_{}".format(i)))
                self.target_agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=critic_state_dim, critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=critic_other_action_dim, maddpg=True,
                                          model_path=model_dir + "_target_adv_{}".format(i)))
            else:  # DDPG
                self.agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=actor_input_dim[i], critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=None, maddpg=False,
                                          model_path=model_dir + "_{}".format(i)))
                self.target_agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=actor_input_dim[i], critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=None, maddpg=False,
                                          model_path=model_dir + "_{}".format(i)))
        self.model_dir = model_dir

    @classmethod
    def init_from_env(cls, env, model_dir):
        """
             Instantiate instance of this class from multi-agent environment
        """
        critic_state_dim, critic_action_dim = 0, 0
        output_dim = []
        get_shape = lambda x: x.shape[0]
        input_dim = []
        type_of_agents = ['adversary' if a.adversary else 'normal' for a in env.agents]
        for action_space, obs_space in zip(env.action_space, env.observation_space):
            input_dim.append(obs_space.shape[0])
            get_shape = lambda x: x.n
            output_dim.append(get_shape(action_space))
            critic_state_dim += input_dim[-1]
            # --------- MADDPG for critic ---------
        for action_space in env.action_space:
            critic_action_dim += get_shape(action_space)
        instance = cls(actor_input_dim=input_dim, type_of_agents=type_of_agents, actor_output_dim=output_dim,
                       critic_state_dim=critic_state_dim, critic_action_dim=critic_action_dim,
                       model_dir=model_dir)
        return instance

    def get_agents(self):
        return self.agents

    def get_target_agents(self):
        return self.target_agents

    def get_num_of_agents(self):
        return self.num_of_agents

    def get_type_of_agents(self):
        return self.type_of_agents

    def step(self, observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.act(obs) for a, obs in zip(self.agents, observations)]

    def get_other_agents_actions(self, states, curr_agent_index, target=False):
        agents = self.agents if not target else self.target_agents
        action_one_hot = []
        for i, agent in enumerate(agents):
            if i == curr_agent_index:
                continue
            agent_i_action = agent.act(state=states[i])
            action_one_hot.append(agent_i_action)
        return action_one_hot

    def update(self, ER, epoch: int, config=cfg):
        """
        wrapper for update_agent, basically preforms update_agent for each agent for the same experiences,
        :param batch_of_samples: batch of samples for each agent
        :param logger:logger handler
        :param epoch:current epoch - for learning rate scheduler
        :param config: hyper params and more
        :return mean of the actor/critic loss
        """
        critic_loss_value_mean = 0

        # this is not according to the paper but it forces the predators to learn a strategy together and not let the
        # smartest predator do all the work
        # (on several occasions just one predator ran after the prey which doesn't work
        # once the prey learns he can just speed up and avoid the predator)
        for i in range(self.num_of_agents):
            batch_of_samples = ER.getMiniBatch(batch_size=config.batch_size)
            # if self.type_of_agents[i] != "adversary":
                # adversary's are the predators, the prey should not learn with the predators,
                # more similar to paper this way
                # batch_of_samples = ER.getMiniBatch(batch_size=config.batch_size)
                # continue
            critic_loss_value = self.update_agent(batch_of_samples=batch_of_samples, agent_index=i,
                                                                  epoch=epoch, config=config)
            critic_loss_value_mean += critic_loss_value
        self.update_targets()  # update targets
        return critic_loss_value_mean / self.num_of_agents

    def update_agent(self, batch_of_samples, agent_index, epoch, config=cfg):
        states, actions, rewards, next_states, dones = batch_of_samples
        curr_agent = self.agents[agent_index]  # get current agent
        target_curr_agent = self.target_agents[agent_index]  # get current target agent
        is_maddpg = curr_agent.is_maddpg()
        #     --------------------- critic train phase ---------------------
        curr_target_agent_action = np.squeeze(target_curr_agent.act(next_states[agent_index]))
        # get target agent one_hot action for next state
        if is_maddpg:  # MADDPG
            # get other target agents actions w.r.t next states
            other_target_agent_actions = self.get_other_agents_actions(states=next_states, curr_agent_index=agent_index,
                                                                       target=True)
            # reshape to two dimensional
            critic_next_state_in = reshape_input_for_critic(next_states)
            critic_other_action_in = reshape_input_for_critic(other_target_agent_actions)
        else:  # DDPG
            critic_other_action_in = None
            critic_next_state_in = next_states[agent_index]
        target_critic_next_state_val = target_curr_agent.get_critic_value(state=critic_next_state_in,
                                                                          action=curr_target_agent_action,
                                                                          other_action=critic_other_action_in)

        # evaluate the target value to train the critic
        is_state_terminal = np.expand_dims((np.ones_like(dones[agent_index]) - np.asarray(dones[agent_index], dtype=np.int)).T, axis=-1)
        critic_target_value = np.reshape(rewards[agent_index], np.shape(target_critic_next_state_val)) + \
                              config.gamma * target_critic_next_state_val * is_state_terminal  # bellman equation
        other_agents_actions = []
        if is_maddpg:  # MADDPG
            critic_state_in = reshape_input_for_critic(states)
            for i, action in enumerate(actions):
                if i == agent_index: continue
                # other_agents_actions = np.concatenate((other_agents_actions, action), axis=-1)\
                other_agents_actions.append(action)
            other_agents_actions = reshape_input_for_critic(other_agents_actions)
        else:  # DDPG
            critic_state_in = states[agent_index]
            other_agents_actions = None

        critic_loss = curr_agent.train_critic(state=critic_state_in, other_action=other_agents_actions,
                                              action=actions[agent_index],
                                              learning_rate=cfg.learning_rate_schedule(epoch),
                                              target=critic_target_value)
        #     --------------------- actor train phase ---------------------
        if is_maddpg:
            other_agents_actions = self.get_other_agents_actions(states=states, curr_agent_index=agent_index)
            other_agents_actions = reshape_input_for_critic(other_agents_actions)
        # TODO maybe try using the given actions... currently following pytorch implementation,
        #  if this is commented out this means we tried, its also according to the paper
        else:
            other_agents_actions = None
        curr_agent.train_actor(actor_state=states[agent_index], critic_state=critic_state_in,
                               other_action=other_agents_actions, learning_rate=cfg.learning_rate_schedule(epoch))
        return critic_loss

    def save(self, epoch, path=None):
        i = 0
        for agent, type in zip(self.agents, self.type_of_agents):
            if path:
                path_i = path + "/predator_{}_{}".format(type, i)
                agent.save(epoch=epoch, path=path_i)
            else:
                agent.save(epoch=epoch)
            i += 1

    def load(self, epoch, path=None):
        i = 0
        for agent, type in zip(self.agents, self.type_of_agents):
            if path:
                path_i = path + "/predator_{}_{}".format(type, i)
                agent.load(epoch=epoch, path=path_i)

            else:
                agent.load(epoch=epoch, path=path)
            i += 1


    def load_normal(self, epoch):
        for i, type in enumerate(self.type_of_agents):
            if type != 'adversary':
                  self.agents[i].load(epoch=epoch)

    def load_adv(self, epoch):
        for i, type in enumerate(self.type_of_agents):
            if type == 'adversary':
                  self.agents[i].load(epoch=epoch)

    def update_targets(self):
        for curr_agent, target_curr_agent in zip(self.agents, self.target_agents):
            actor_params, critic_params = curr_agent.get_weights()
            target_curr_agent.soft_update(params=actor_params, actor_flag=True)
            target_curr_agent.soft_update(params=critic_params, actor_flag=False)


class PredatorStudent(StudentActor):

    def _get_initial_size(self):
        return [self.actor_input_dim[-1] * cfg.hidden_dim, cfg.hidden_dim ** 2,
                cfg.hidden_dim * self.actor_output_dim[-1]]

    def _calculate_sizes_according_to_redundancy(self):
        assert self.redundancy is not None
        initial_sizes = self._get_initial_size()
        total_number_of_parameters_for_next_iteration = []
        for i, initial_num_of_params_at_layer in enumerate(initial_sizes):
            total_number_of_parameters_for_next_iteration.append(
                int(initial_num_of_params_at_layer * (1 - self.redundancy[i])))  # 1 - whats not important = important
        current_size = total_number_of_parameters_for_next_iteration[0] * self.actor_input_dim[-1] + \
                       total_number_of_parameters_for_next_iteration[1] * total_number_of_parameters_for_next_iteration[0] + \
                       self.actor_output_dim[-1] * total_number_of_parameters_for_next_iteration[1]
        # to ensure that the size is monotonically decreasing
        while current_size > self.last_measure:
            i = np.argmax(total_number_of_parameters_for_next_iteration)
            total_number_of_parameters_for_next_iteration[i] -= 1
            current_size = total_number_of_parameters_for_next_iteration[0] * self.actor_input_dim[-1] + \
                           total_number_of_parameters_for_next_iteration[1] * total_number_of_parameters_for_next_iteration[0] + \
                           self.actor_output_dim[-1] * total_number_of_parameters_for_next_iteration[1]

        # to ensure no zero parameters
        i = np.argmin(total_number_of_parameters_for_next_iteration)
        while total_number_of_parameters_for_next_iteration[i] < 1:
            total_number_of_parameters_for_next_iteration[i] = 4
            total_number_of_parameters_for_next_iteration[0] = (total_number_of_parameters_for_next_iteration[0] +
                                          total_number_of_parameters_for_next_iteration[1]) / (
                                                 self.actor_input_dim[-1] + total_number_of_parameters_for_next_iteration[1])
            i = np.argmin(total_number_of_parameters_for_next_iteration)

        for i, size in enumerate(total_number_of_parameters_for_next_iteration):
            total_number_of_parameters_for_next_iteration[i] = int(size)

        return total_number_of_parameters_for_next_iteration

    def _build_actor(self):
        if self.redundancy is None:
            fc_1_dim = cfg.hidden_dim
            fc_2_dim = cfg.hidden_dim
        else:
            new_size_parameters = self._calculate_sizes_according_to_redundancy()
            fc_1_dim = new_size_parameters[0]
            fc_2_dim = new_size_parameters[1]
        with tf.variable_scope('actor'):
            input = tf.keras.layers.BatchNormalization()(self.actor_state)
            with tf.variable_scope("fc_1") as scope_tf:
                fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.actor_input_dim[-1], fc_1_dim), activation=tf.nn.relu)
            with tf.variable_scope("fc_2") as scope_tf:
                fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(fc_1_dim, fc_2_dim), activation=tf.nn.relu)

            with tf.variable_scope("fc_3") as scope_tf:
                fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
                                       weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                       shape=(fc_2_dim, self.actor_output_dim[-1]))
            return fc_3


class StudentMaddpgWrapper(MaddpgWrapper):
    def __init__(self, actor_input_dim,
                 type_of_agents,
                 actor_output_dim,
                 critic_state_dim,
                 critic_action_dim,
                 student_index,
                 model_dir,
                 student_redundancy=None,
                 last_measure=1e5):
        """
        Initialize an instance of the MADDPG model with one soon-to-be-pruned student
        :param actor_input_dim:
        :param actor_output_dim:
        :param critic_state_dim:
        :param critic_action_dim:
         :param student_index: student agent index
        :param model_dir:
        """
        self.student_index = student_index
        self.num_of_agents = len(type_of_agents)
        self.agents = []
        self.type_of_agents = type_of_agents
        for i, type in enumerate(type_of_agents):
            if i == student_index:
                self.agents.append(PredatorStudent(actor_input_dim,
                 actor_output_dim, model_dir + "/pruned_student/", redundancy=student_redundancy, last_measure=last_measure))

            if type == "adversary":  # MADDPG
                critic_other_action_dim = critic_action_dim - actor_output_dim[i]
                self.agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=critic_state_dim, critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=critic_other_action_dim, maddpg=True,
                                          model_path=model_dir + "_adv_{}".format(i)))
            else:  # DDPG
                self.agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=actor_input_dim[i], critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=None, maddpg=False,
                                          model_path=model_dir + "_{}".format(i)))
        self.model_dir = model_dir



    def step(self, observations):
        """
           Take a step forward in environment with all agents
           Inputs:
               observations: List of observations for each agent
               explore (boolean): Whether or not to add exploration noise
           Outputs:
               actions: List of actions for each agent
               teacher_Q_value:  student_index_Q_value
           """
        action = []
        for agent, obs, i in zip(self.agents, observations, range(len(self.agents))):
                if i == self.student_index:
                    teacher_Q_value = agent.get_q(obs)

        return [a.act(obs) for a, obs in zip(self.agents, observations)]




