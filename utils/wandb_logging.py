import wandb
from time import localtime, strftime


class WAndB():
    def __init__(self, config, run_name='', project="lux-ai-2022", entity="uib-lux-ai"):

        current_time = strftime("%Y-%m-%d %H:%M", localtime())

        wandb.init(config=config, project=project, entity=entity)
        wandb.run.name = run_name if run_name != '' else f'Untitled run at: {current_time}'

    def log(self, data: dict, step = None):
        wandb.log(data, step = step, commit=True)
