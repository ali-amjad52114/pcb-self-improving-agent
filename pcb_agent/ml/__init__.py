"""ML side of the agent: the training config space and the trainer.

`simulator.Trainer` is a stand-in that satisfies the frozen interfaces from the
3-person plan (`train_model`, `evaluate_model`, `apply_experiment`) so the
Fireworks loop is runnable before the real PCB pipeline exists. Replace
`Trainer` with the real thing and nothing upstream changes.

`apply_experiment` lives in `pcb_agent.fireworks.tools`, because the mutation
has to agree exactly with the tool schemas the model chooses from.
"""

from pcb_agent.fireworks.tools import DEFAULT_TRAINING_CONFIG, apply_experiment

from .simulator import PCB_CLASSES, TASK_A, TASK_B, SimulatedTask, Trainer

__all__ = [
    "Trainer",
    "SimulatedTask",
    "TASK_A",
    "TASK_B",
    "PCB_CLASSES",
    "DEFAULT_TRAINING_CONFIG",
    "apply_experiment",
]
