class Environment:
    def __init__(self):
        self.state = "initial"

    def update(self, action):
        self.state = f"updated with {action}"
        return self.state
