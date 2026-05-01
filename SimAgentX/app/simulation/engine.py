from app.agent.agent import Agent
from app.environment.environment import Environment

class SimulationEngine:
    def __init__(self):
        self.agent = Agent()
        self.env = Environment()

    def run(self):
        state = self.env.state
        action = self.agent.decide(state)
        new_state = self.env.update(action)
        print(f"Agent action: {action}")
        print(f"New state: {new_state}")
