from environment import GridWorldEnv
from dqn_agent import DQNAgent


def train_agent():
    env = GridWorldEnv(grid_size=6)

    agent = DQNAgent(
        state_dim=2,
        action_dim=4
    )

    episodes = 300

    for ep in range(episodes):

        state = env.reset()
        total_reward = 0

        for step in range(100):

            action = agent.act(state)

            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")


if __name__ == "__main__":
    train_agent()
