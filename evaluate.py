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


def evaluate(env, maddpg: MaddpgWrapper, cfg=config):
    type_of_agents = maddpg.get_type_of_agents()
    num_of_agents = maddpg.get_num_of_agents()
    reward_mean = np.zeros(num_of_agents)
    collisons_mean = []
    total_collisons_episode = 0
    for e in range(100):
        state = env.reset()
        collisons_per_epoch = np.zeros(4)
        reward_per_episode = np.zeros(num_of_agents)
        for time_step in range(250):
            # env.render('')
            # sleep(0.05)
            actions = maddpg.step(observations=reshape_state(state))
            next_state, rewards, dones, collisons = env.step(actions)
            collisons_per_epoch += collisons['n']
            reward_per_episode += rewards
            state = next_state
            if all(dones):
                break
        reward_mean += reward_per_episode
        collisons_mean.append(collisons_per_epoch)
        total_collisons_episode += np.sum(collisons_per_epoch)
        for reward_per_agent, collisions_i, type, index in zip(reward_per_episode, collisons_per_epoch, type_of_agents, range(num_of_agents)):
            print("Episode {}/{} : average reward for {} agent {} is: \t {}"
                  .format(e, 100, type, index, reward_per_agent))
            print("Episode {}/{} :  collisons for {} agent {} is: \t {}"
                  .format(e, 100, type, index, collisions_i))
    total_collisons_episode /= 100
    reward_mean /= 100
    collisons_mean = np.mean(collisons_mean, axis=0)
    print("--------------- Evaluation Report --------------- ")
    for reward_per_agent, type, collisions_i,  index in zip(reward_mean, type_of_agents, collisons_mean, range(num_of_agents)):
        print("Evaluation: Reward for {} agent {} is: \t {}"
              .format(type, index, reward_per_agent))
        print("Evaluation: Collisons for {} agent {} is: \t {}"
              .format(type, index, collisions_i))
    print("Total collision average : {}".format(total_collisons_episode))
    maddpg.save(epoch='ready', path=cfg.ready_model)


def make_env(scenario_name, benchmark=False):
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
    env = make_env('simple_tag', benchmark=True)
    maddpg = MaddpgWrapper.init_from_env(env, model_dir=config.model_dir)
    # maddpg.load_adv(epoch="99000")
    # maddpg.load_normal(epoch="54000")
    maddpg.load(epoch="99000")
    evaluate(env, maddpg)
    a = 5






