from utilss.tensorflow_utils import _build_fc_layer, reshape_input_for_critic, transform_one_hot_batch, transform_one_hot
import tensorflow as tf
import numpy as np
from tensorflow.contrib.model_pruning.python import pruning
from abc import abstractmethod
import os
from config import DensePredatorAgentConfig as cfg
from gym.spaces import Box, Discrete
from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import RelaxedOneHotCategorical, ExpRelaxedOneHotCategorical
import random
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
                 critic_other_action_dim, model_path, hidden_dim=cfg.hidden_dim, tau=cfg.tau, maddpg=True):
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
        self.hidden_dim = hidden_dim
        self.tau = tau
        with self.graph.as_default():
                self.critic_global_step = tf.Variable(0, trainable=False)
                self.actor_global_step = tf.Variable(0, trainable=False)
                self._build_placeholders()
                self.actor_logits = self._build_actor()
                self.gumbel_dist = self._build_gumbel(self.actor_logits)
                self.critic_logits = self._build_critic(self.critic_action)
                self.actor_weight_matrices = tf.get_collection(pruning._MASKED_WEIGHT_COLLECTION)
                self.actor_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
                self.critic_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
                self.critic_loss = self._build_critic_loss()
                self.critic_train_op = self._build_critic_train_op()
                self.actor_train_op = self._build_actor_train_op()
                self.actor_saver = tf.train.Saver(var_list=self.actor_parameters, max_to_keep=100)
                self.critic_saver = tf.train.Saver(var_list=self.critic_parameters, max_to_keep=100)
                self.init_variables(tf.global_variables())
                self.actor_soft_update, self.actor_target_placeholders = self._build_soft_sync_op(self.actor_parameters)
                self.critic_soft_update, self.critic_target_placeholders = self._build_soft_sync_op(self.critic_parameters)

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
        actor_logits = self.sess.run(self.actor_logits, feed_dict={self.actor_state: state})
        action = np.argmax(actor_logits, axis=-1)
        batch_size = np.shape(action)[0]
        if batch_size == 1:
            return transform_one_hot(action, self.actor_output_dim[-1])
        else:
            return transform_one_hot_batch(action, self.actor_output_dim[-1])

    def get_gumbel(self, state):
        """
        get an action in one_hot format according to a gumbel distribution
        :param state: state
        :param temperature: low temperature outputs a one_hot action distribution, high temperature outputs
                            a uniform distribution (intuitively-high temperature meanse lower weights to the logits)
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
        # self.temperature = tf.placeholder_with_default(1., shape=None, name='temperature')

    def _build_actor(self, scope='actor'):
        with tf.variable_scope(scope):
            input = tf.keras.layers.BatchNormalization()(self.actor_state)
            with tf.variable_scope("fc_1") as scope_tf:
                fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.actor_input_dim[-1], self.hidden_dim), activation=tf.nn.relu)
            with tf.variable_scope("fc_2") as scope_tf:
                fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.hidden_dim, self.hidden_dim), activation=tf.nn.relu)

            with tf.variable_scope("fc_3") as scope_tf:
                 fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
                                       weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                       shape=(self.hidden_dim, self.actor_output_dim[-1]))
            return fc_3

    def _build_critic(self, critic_action, scope='critic', reuse=False):
        with tf.variable_scope(scope) as scope_critic:
            if reuse:
                scope_critic.reuse_variables()
            if self.maddpg:
                critic_input = tf.concat([self.critic_state, critic_action, self.critic_other_action], axis=-1)
            else:
                critic_input = tf.concat([self.critic_state, critic_action], axis=-1)
            input = tf.keras.layers.BatchNormalization()(critic_input)
            with tf.variable_scope("fc_1") as scope_tf:
                fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.critic_input_dim, self.hidden_dim), activation=tf.nn.relu)
            with tf.variable_scope("fc_2") as scope_tf:
                fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
                                       weight_init=tf.keras.initializers.glorot_uniform(),
                                       shape=(self.hidden_dim, self.hidden_dim), activation=tf.nn.relu)

            with tf.variable_scope("fc_3") as scope_tf:
                fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
                                            weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                            shape=(self.hidden_dim, 1))
            return fc_3

    def _build_critic_loss(self):
        """
        build loss function = MSE(target, logits)
        :return: a loss fn
        """
        mse = tf.losses.mean_squared_error(labels=self.critic_target,
                                           predictions=self.critic_logits)
        # mse = tf.reduce_mean(mse)
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
        return gumbel_softmax(logits=actor_logits)

    def _build_soft_sync_op(self, network_params):
                params = self.sess.run(network_params)  # DNN weights

                target_network_parameters = [tf.placeholder(dtype=tf.float32, shape=np.shape(network_parameter))
                                                  for network_parameter in params]  # new weights
                # new weights are weighted with tau
                return [tf.assign(parameter, (1 - self.tau) * parameter + self.tau * weight)
                                    for parameter, weight in zip(network_params,  target_network_parameters)], target_network_parameters


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
                                          model_path=model_dir + "adv_{}".format(i)))
                self.target_agents.append(MADDPG(actor_input_dim=actor_input_dim[i], actor_output_dim=actor_output_dim[i],
                                          critic_state_dim=critic_state_dim, critic_action_dim=actor_output_dim[i],
                                          critic_other_action_dim=critic_other_action_dim, maddpg=True,
                                          model_path=model_dir + "target_adv_{}".format(i)))
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
        discrete_action = True
        output_dim = []
        get_shape = lambda x: x.shape[0]
        input_dim = []
        type_of_agents = ['adversary' if a.adversary else 'normal' for a in env.agents]
        for action_space, obs_space in zip(env.action_space, env.observation_space):
            input_dim.append(obs_space.shape[0])
            if isinstance(action_space, Box):
                discrete_action = False
            else:  # Discrete
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
        for i in range(self.num_of_agents):
            batch_of_samples = ER.getMiniBatch(batch_size=config.batch_size)
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
            # TODO maybe try using the given actions... currently following pytorch implementation
        else:
            other_agents_actions = None
        curr_agent.train_actor(actor_state=states[agent_index], critic_state=critic_state_in,
                               other_action=other_agents_actions, learning_rate=cfg.learning_rate_schedule(epoch))
        return critic_loss

    def save(self, epoch):
        for agent in self.agents:
            agent.save(epoch=epoch)

    def load(self, epoch):
        for agent in self.agents:
            agent.load(epoch=epoch)

    def update_targets(self):
        for curr_agent, target_curr_agent in zip(self.agents, self.target_agents):
            actor_params, critic_params = curr_agent.get_weights()
            target_curr_agent.soft_update(params=actor_params, actor_flag=True)
            target_curr_agent.soft_update(params=critic_params, actor_flag=False)



# class Network(BaseNetwork):
#     def __init__(self, input_dim, out_dim, discrete_action, model_path, scope):
#         super(Network, self).__init__(model_path)
#         self.input_dim = (None, input_dim)
#         self.out_dim = (None, out_dim)
#         self.discrete_action = discrete_action
#         with self.graph.as_default():
#             self.global_step = tf.Variable(0, trainable=False)
#             self._build_placeholders()
#             self.logits = self._build_logits(scope)
#             self.weights_matrices = pruning.get_masked_weights()
#             self.loss = self._build_loss()
#             self.train_op = self._build_train_op()
#             self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
#             self.init_variables(tf.global_variables())
#
#     def _build_placeholders(self):
#         self.input = tf.placeholder(dtype=tf.float32, shape=self.input_dim, name='input')
#         self.target = tf.placeholder(dtype=tf.float32, shape=self.out_dim, name='target')
#         self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='learning_rate')
#         self.error_weights = tf.placeholder(dtype=tf.float32, shape=None, name='td_errors_weight')
#
#     def get_q(self, state):
#         """
#         feed the state through the network
#         :param state: input to DNN
#         :return: output vector with out_dim elements
#         """
#         return self.sess.run(self.logits, feed_dict={self.input: state})
#
#     def soft_update(self, params):
#         with self.graph.as_default():
#             params_dict = {}
#             for place_holder, param in zip(self.target_network_parameters, params):
#                 params_dict[place_holder] = param
#             self.sess.run(self.soft_sync, feed_dict=params_dict)
#
#     def get_weights(self):
#         with self.graph.as_default():
#             return self.sess.run(self.network_parameters)
#
#     def get_params(self):
#         with self.graph.as_default():
#             return self.sess.run(self.network_parameters)
#
#     @abstractmethod
#     def _build_logits(self, scope):
#         pass
#
#     @abstractmethod
#     def _build_loss(self):
#         self.td_errors = None
#         pass
#
#     @abstractmethod
#     def _build_train_op(self):
#         pass
#
#
# class ActorMLPNetwork(Network):
#     """
#     MLP network (can be used as value or policy)
#     """
#     def __init__(self, input_dim, out_dim, model_path, scope, discrete_action=True, tau=0.01, hidden_dim=64):
#
#         self.hidden_dim = hidden_dim
#         self.tau = tau
#         super(ActorMLPNetwork, self).__init__(input_dim, out_dim, discrete_action, model_path, scope=scope)
#         with self.graph.as_default():
#             self.network_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#             params = self.sess.run(self.network_parameters)
#             self.target_network_parameters = [tf.placeholder(dtype=tf.float32, shape=np.shape(network_parameter))
#                                               for network_parameter in params]
#             self.soft_sync = [tf.assign(parameter, (1 - tau) * parameter + tau * weight)  # self is target
#                                 for parameter, weight in zip(self.network_parameters,  self.target_network_parameters)]
#
#     def _build_placeholders(self):
#         super(ActorMLPNetwork, self)._build_placeholders()
#         # self.gumbel = tf.placeholder(tf.bool, shape=[])
#         self.temperature = tf.placeholder_with_default(input=1.0, shape=[])
#         self.action_gradient_from_critic = tf.placeholder(dtype=tf.float32, shape=self.out_dim,
#                                                           name='grad_from_critic')
#         self.batch_size = tf.placeholder(dtype=tf.float32, shape=None, name='batch_size')
#
#     # def soft_update(self, params):
#     #     with self.graph.as_default():
#     #         params_dict = {}
#     #         for place_holder, param in zip(self.target_network_parameters, params):
#     #             params_dict[place_holder] = param
#     #         self.sess.run(self.soft_sync, feed_dict=params_dict)
#     #
#     # def get_params(self):
#     #     with self.graph.as_default():
#     #         return self.sess.run(self.network_parameters)
#     #
#     # def get_weights(self):
#     #     with self.graph.as_default():
#     #         return self.sess.run(self.network_parameters)
#
#     def get_q(self, state, temperature=1.0):
#         """
#         feed the state through the network
#         :param state: input to DNN
#         :param temperature: temperature to gumbel distribution
#         :return: output vector with out_dim elements
#         """
#         return self.sess.run(self.logits, feed_dict={self.input: state, self.temperature: temperature})
#
#     def act(self, state):
#         """
#         feed the state through the network
#         :param state: input to DNN
#         :param temperature: temperature to gumbel distribution
#         :return: output vector with out_dim elements
#         """
#         logits = self.sess.run(self.fc_3, feed_dict={self.input: state})
#         action = np.argmax(logits, axis=-1)
#         action = transform_one_hot_batch(batch_index=action, size=self.out_dim[-1])
#         return np.squeeze(action)
#
#     def _build_logits(self, scope):
#         with tf.variable_scope(scope):
#             input = tf.keras.layers.BatchNormalization()(self.input)
#             with tf.variable_scope("fc_1") as scope_tf:
#                 fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
#                                        weight_init=tf.keras.initializers.glorot_uniform(),
#                                        shape=(self.input_dim[-1], self.hidden_dim), activation=tf.nn.relu)
#             with tf.variable_scope("fc_2") as scope_tf:
#                 fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
#                                        weight_init=tf.keras.initializers.glorot_uniform(),
#                                        shape=(self.hidden_dim, self.hidden_dim), activation=tf.nn.relu)
#
#             with tf.variable_scope("fc_3") as scope_tf:
#                  self.fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
#                                        weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
#                                        shape=(self.hidden_dim, self.out_dim[-1]))
#             output = RelaxedOneHotCategorical(temperature=self.temperature, logits=self.fc_3).sample()
#             return output
#
#     def _build_train_op(self):
#         """
#          build Optimizer in graph
#         :return: an optimizer
#         """
#         network_params = tf.trainable_variables()
#         # batch_size = tf.dtypes.cast(tf.shape(self.input)[0], tf.float32)
#         self.gradients = tf.gradients(ys=self.logits, xs=network_params,
#                                  grad_ys=-self.action_gradient_from_critic / self.batch_size)
#         gradients = self.gradients
#         # we are doing gradient ascent like the DDPG gradient purposed by David Silver
#         # the gradient of the actor parameters w.r.t the state and the action chosen, is the critic gradient w.r.t
#         # to the action chosen multiplied with the gradient of the action chosen w.r.t to the actors parameters
#         # derivative of ys w.r.t to xs is the gradient of the action w.r.t to the a_params multiplied by grad_ys, which
#         # is the gradient from the critic of the state and the action w.r.t to the critic params
#         # capped_gvs = []
#         # for grad in gradients:  # clip gradients
#         #     capped_gvs.append(tf.clip_by_value(grad, 0.5))
#         return tf.train.AdamOptimizer(self.learning_rate). \
#             apply_gradients(zip(gradients, network_params), global_step=self.global_step)
#
#     def train_actor(self, actor_state_in, q_gradient_input, learning_rate):
#         batch_size = np.shape(actor_state_in)[0]
#         _, gradients = self.sess.run([self.train_op, self.gradients],
#                       feed_dict={self.input: actor_state_in,
#                                  self.action_gradient_from_critic: q_gradient_input,
#                                  self.learning_rate: learning_rate,
#                                  self.temperature: 1.0,
#                                  self.batch_size: batch_size})
#
#     # def _build_logits_critic(self, scope):
#     #     with tf.variable_scope(scope):
#     #         with tf.variable_scope("fc_1") as scope_tf:
#     #             fc_1 = _build_fc_layer(self=self, inputs=self.input, scope=scope_tf,
#     #                                    weight_init=tf.keras.initializers.glorot_uniform(),
#     #                                    shape=(self.input_dim[-1], self.hidden_dim), activation=tf.nn.relu)
#     #         with tf.variable_scope("fc_2") as scope_tf:
#     #             fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
#     #                                    weight_init=tf.keras.initializers.glorot_uniform(),
#     #                                    shape=(self.hidden_dim, self.hidden_dim), activation=tf.nn.relu)
#     #
#     #         with tf.variable_scope("fc_3") as scope_tf:
#     #              self.fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
#     #                                    weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
#     #                                    shape=(self.hidden_dim, self.out_dim[-1]))
#     #         return self.fc_3
#
#
#
# class CriticMLPNetwork(Network):
#     """
#     MLP network (can be used as value or policy)
#     """
#
#     def __init__(self, state_in_dim, action_in_dim, out_dim, model_path, scope, discrete_action=True, tau=0.01, hidden_dim=64):
#
#         self.hidden_dim = hidden_dim
#         self.tau = tau
#         self.action_in_dim = action_in_dim
#         self.state_dim = state_in_dim
#         super(CriticMLPNetwork, self).__init__(state_in_dim + action_in_dim, out_dim, discrete_action, model_path, scope=scope)
#         with self.graph.as_default():
#             self.network_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#             params = self.sess.run(self.network_parameters)
#             self.target_network_parameters = [tf.placeholder(dtype=tf.float32, shape=np.shape(network_parameter))
#                                               for network_parameter in params]
#             self.soft_sync = [tf.assign(parameter, (1 - tau) * parameter + tau * weight)  # self is target
#                               for parameter, weight in zip(self.network_parameters, self.target_network_parameters)]
#             self.actor_gradients = tf.gradients(ys=self.logits, xs=self.action_in)
#
#     def compute_gradients(self, state, action):
#         return self.sess.run([self.actor_gradients, self.logits], feed_dict={self.state_in: state,
#                                                                              self.action_in: action})
#
#     def _build_placeholders(self):
#         super(CriticMLPNetwork, self)._build_placeholders()
#         # self.gumbel = tf.placeholder(tf.bool, shape=[])
#         self.action_in = tf.placeholder(dtype=tf.float32, shape=(None, self.action_in_dim), name='input_action')
#         self.state_in = tf.placeholder(dtype=tf.float32, shape=(None, self.state_dim), name='input_state')
#         self.input = tf.concat(axis=1, values=[self.state_in, self.action_in])
#
#     def get_q(self, state):
#         """
#         feed the state through the network
#         :param state: input to DNN
#         :return: output vector with out_dim elements
#         """
#         return self.sess.run(self.logits, feed_dict={self.input: state})
#
#     def _build_logits(self, scope):
#         with tf.variable_scope(scope):
#             input = tf.keras.layers.BatchNormalization()(self.input)
#             with tf.variable_scope("fc_1") as scope_tf:
#                 fc_1 = _build_fc_layer(self=self, inputs=input, scope=scope_tf,
#                                        weight_init=tf.keras.initializers.glorot_uniform(),
#                                        shape=(self.input_dim[-1], self.hidden_dim), activation=tf.nn.relu)
#             with tf.variable_scope("fc_2") as scope_tf:
#                 fc_2 = _build_fc_layer(self=self, inputs=fc_1, scope=scope_tf,
#                                        weight_init=tf.keras.initializers.glorot_uniform(),
#                                        shape=(self.hidden_dim, self.hidden_dim), activation=tf.nn.relu)
#
#             with tf.variable_scope("fc_3") as scope_tf:
#                 self.fc_3 = _build_fc_layer(self=self, inputs=fc_2, scope=scope_tf,
#                                             weight_init=tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
#                                             shape=(self.hidden_dim, self.out_dim[-1]))
#             return self.fc_3
#
#     def _build_train_op(self):
#         """
#          build Optimizer in graph
#         :return: an optimizer
#         """
#         # return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
#         #                                                                          global_step=self.global_step)
#         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
#         gradients = optimizer.compute_gradients(loss=self.loss, var_list=tf.trainable_variables())
#         # capped_gvs = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in gradients]
#         return optimizer.apply_gradients(gradients)
#
#     def _build_loss(self):
#         """
#         build loss function = MSE(target, logits)
#         :return: a loss fn
#         """
#         mse = tf.losses.mean_squared_error(labels=self.target,
#                                            predictions=self.logits)  # if not per then this error weights are 1
#         mse = tf.reduce_mean(mse)
#         return mse
#
#     def learn(self, target_batch, learning_rate, input):
#         """
#         one batch learning function via gradient decent method
#         :param target_batch: target batch, essential for loss function calculation
#         :param learning_rate: learning rate for determining the step size in the iterative gradient decent method
#         :param input: input to DNN
#         :param weights: weights for elements in batch ( used to weight the loss function according to the elements)
#                         needed only when using Prioritized experience replay
#         :return: mean loss value and the td error for each element in the batch w.r.t to according element in Model(input)
#         """
#         _, loss = self.sess.run([self.train_op, self.loss],
#                                            feed_dict={self.input: input,
#                                                       self.target: target_batch,
#                                                       self.learning_rate: learning_rate})
#         return loss


# class DDPGAgent:
#
#     def __init__(self, input_dim, output_dim, critic_state_dim, critic_action_dim,
#                  model_dir,
#                  index,
#                  type,
#                  hidden_dim=64,
#                  discrete_action=False,
#                  explore_eps=1.0):
#         with tf.variable_scope('Actor_{}'.format(index)):
#             self.policy = ActorMLPNetwork(input_dim=input_dim, out_dim=output_dim, discrete_action=discrete_action,
#                                           model_path=model_dir + "/actor", hidden_dim=hidden_dim, scope='eval')
#             self.critic = CriticMLPNetwork(state_in_dim=critic_state_dim, action_in_dim=critic_action_dim, out_dim=1,
#                                            discrete_action=True, model_path=model_dir + "/critic",
#                                            hidden_dim=hidden_dim, scope='eval')
#         with tf.variable_scope('Critic_{}'.format(index)):
#             self.target_policy = ActorMLPNetwork(input_dim=input_dim, out_dim=output_dim, discrete_action=discrete_action,
#                                                  model_path=model_dir + "/actor", hidden_dim=hidden_dim, scope='target')
#             self.target_critic = CriticMLPNetwork(state_in_dim=critic_state_dim, action_in_dim=critic_action_dim,  out_dim=1,
#                                                   discrete_action=True, model_path=model_dir + "/critic",
#                                                   hidden_dim=hidden_dim, scope='target')
#         self.discrete_action = discrete_action
#         self.output_dim = output_dim
#         self.input_dim = input_dim
#         self.index = index
#         self.type = type
#         self.explore_eps = explore_eps
#
#     def step(self, state, temp):
#         """
#         Take a step forward in environment for a minibatch of states
#         :param state: Observations for this agent
#         :param explore: Whether or not to add exploration noise
#         :return: Actions for this agent
#         """
#         action_dist = self.policy.get_q(state=state, temperature=temp)
#         action = np.random.choice(np.arange(self.output_dim), p=action_dist.ravel())
#         action = transform_one_hot(index=action, size=self.output_dim)
#         return np.squeeze(action)
#
#     def degrade_explore(self):
#         self.explore_eps = max(0.0001, self.explore_eps * 0.9995)
#         return self.explore_eps
#     # def target_step(self, state, critic_state):
#     #     q_values = self.target_policy.get_q(state=state)
#     #     critic_feedback = self.target_critic.get_q(state=critic_state)
#     #     return q_values, critic_feedback
#
#     def save_model(self, epoch):
#         self.policy.save_model(epoch)
#         self.critic.save_model(epoch)
#
#     def load_model(self, epoch, model_dir):
#             self.policy.load_model(path=model_dir + "/actor", epoch=epoch)
#             self.critic.load_model(path=model_dir + "/critic", epoch=epoch)
#
#     def soft_update(self):
#         critic_params = self.critic.get_params()
#         self.target_critic.soft_update(params=critic_params)
#         actor_params = self.policy.get_params()
#         self.target_policy.soft_update(params=actor_params)
#
#







