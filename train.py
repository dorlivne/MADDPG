import numpy as np
from model import MaddpgWrapper
from utilss.Memory import MAExperienceReplay
from utilss.logger_utils import get_logger

from config import DensePredatorAgentConfig as config
from time import gmtime, strftime, sleep
from torch.utils.tensorboard import SummaryWriter


def reshape_state(state):
    return [np.expand_dims(s, axis=0) for s in state]


def train_on_batch(epoch, maddpg:MaddpgWrapper, ER: MAExperienceReplay, cfg=config):
    # states, actions, rewards, next_states, is_done_vec = ER.getMiniBatch(batch_size=cfg.batch_size)
    # NOTE: For now we use the same experiences for every agent to train on,
    # it might be better to mix it up an sample for each agent
    # on the other hand, it might be useful if all the agents evaluate and better themselves on the same experiences
    # sounds like a joint effort
    critic_loss = maddpg.update(ER, epoch, cfg)  # the critic q value is the value that the actor is trying to maximize
    return critic_loss


def train(logger, env, maddpg: MaddpgWrapper, cfg=config):
    eval_flag = False
    ER = MAExperienceReplay(size=cfg.buffer_length, num_of_agents=maddpg.get_num_of_agents())
    t = 0
    type_of_agents = maddpg.get_type_of_agents()
    num_of_agents = maddpg.get_num_of_agents()
    writer = SummaryWriter(log_dir=cfg.logs_dir + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + '- predator-prey')
    reward_mean = np.zeros(num_of_agents)
    critic_loss_mean = []
    collisions_mean = []
    for e in range(cfg.epochs):
        state = env.reset()
        reward_per_episode = np.zeros(num_of_agents)
        collisions_per_episode = 0
        for time_step in range(50):
            if eval_flag or e % 500 < 2:
                env.render('')
                sleep(0.03)
            t += 1
            actions = maddpg.step(observations=reshape_state(state))
            next_state, rewards, dones, collisions = env.step(actions)
            collisions_per_episode += np.sum(collisions['n'])
            reward_per_episode += rewards
            ER.add_memory(state=state, reward=rewards, action=actions, next_state=next_state, is_done=dones)
            state = next_state
            if not eval_flag and t > cfg.batch_size and (t % cfg.steps_per_update) == 0:  # train freq and start
                critic_loss = train_on_batch(e, maddpg, ER, cfg)  # train phase and update target networks
                critic_loss_mean.append(critic_loss)
            if all(dones):
                break
        collisions_mean.append(collisions_per_episode)
        reward_mean += reward_per_episode
        if (e + 1) % 100 == 0:
            reward_mean /= 100
            critic_loss_mean = np.mean(critic_loss_mean)
            collisions_mean = np.mean(collisions_mean)
            for reward_per_agent, type, index in zip(reward_mean, type_of_agents, range(num_of_agents)):
                logger.info("Episode {}/{} : average reward for {} agent {} is: \t {}"
                            .format(e, cfg.epochs, type, index, reward_per_agent))
                print("Episode {}/{} : average reward for {} agent {} is: \t {}"
                      .format(e, cfg.epochs, type, index, reward_per_agent))
                writer.add_scalar("{}_agent_{}/Reward Agent".format(type, index), reward_per_agent, e)
            print("Episode {}/{} : average collisions {} critic loss value {}".format(e, cfg.epochs, collisions_mean, critic_loss_mean))
            logger.info("Episode {}/{} : average collisions {} critic loss value {}".format(e, cfg.epochs, collisions_mean, critic_loss_mean))
            reward_mean = np.zeros(num_of_agents)
            critic_loss_mean = []
            writer.add_scalar("predators/Collisions", collisions_mean, e)
            if collisions_mean >= cfg.environment_solved_objective and not eval_flag:
                print("------------  Episode {}/{} : environment solved with score {} over 2000 continuous episdoes "
                      "------------".format(e, cfg.epochs, collisions_mean))
                logger.info("------------  Episode {}/{} : environment solved with score {} over 2000 continuous episdoes"
                            " ------------".format(e, cfg.epochs, collisions_mean))
                maddpg.save("best_{}".format(e))
            collisions_mean = []
        if e % cfg.save_interval == 0 and not eval_flag:
            maddpg.save(epoch=e)


def make_env(scenario_name, benchmark=True):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == '__main__':
    logger = get_logger('predator_prey_log')
    env = make_env('simple_tag')
    maddpg = MaddpgWrapper.init_from_env(env, model_dir=config.model_dir)
    # maddpg.load(epoch="99000")
    train(logger, env, maddpg)
    a = 5






