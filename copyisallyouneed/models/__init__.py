from .agent import Agent
from .copyisallyouneed import Copyisallyouneed

def load_model(args):
    model = Copyisallyouneed(**args)
    agent = Agent(model, args)
    return agent
