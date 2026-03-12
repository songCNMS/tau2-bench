"""
Base classes for task synthesis, following the telecom domain pattern.

The synthesis architecture mirrors the telecom TaskManager/BaseTask/SelectionSet
pattern but adapted for domains that operate on a shared database (airline, retail)
rather than on separate agent/user environments.

Key abstractions:
- ScenarioParam: A parameterized scenario component (e.g., a specific user + reservation)
- ScenarioTemplate: A template that, given params, produces a complete Task
- TaskSynthesizer: Generates tasks by combining templates with sampled params
"""

import json
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from tau2.data_model.tasks import (
    Action,
    Description,
    EvaluationCriteria,
    RewardType,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.environment.environment import Environment


class ScenarioParam(BaseModel):
    """Parameters extracted from the database that instantiate a scenario template."""

    user_id: str
    user_name: str  # "First Last"
    known_info: str  # What the user knows about themselves
    extra: dict[str, Any] = Field(default_factory=dict)  # Domain-specific params


class ScenarioOutcome(BaseModel):
    """The expected outcome of a scenario: golden actions + evaluation criteria."""

    actions: list[Action] = Field(default_factory=list)
    nl_assertions: list[str] = Field(default_factory=list)
    communicate_info: list[str] = Field(default_factory=list)
    reward_basis: list[RewardType] = Field(
        default_factory=lambda: [RewardType.DB, RewardType.COMMUNICATE]
    )


class ScenarioTemplate(ABC):
    """
    A template for generating task scenarios.

    Each template represents a class of tasks (e.g., "cancel eligible reservation",
    "exchange delivered items"). It:
    1. Samples valid parameters from the database
    2. Produces a complete Task with user scenario + evaluation criteria
    3. Self-verifies by running golden actions against a fresh environment
    """

    name: str
    domain: str
    category: str  # e.g., "cancellation", "exchange", "refusal"

    @abstractmethod
    def sample_params(
        self, env: Environment, rng: random.Random
    ) -> Optional[ScenarioParam]:
        """
        Sample valid parameters from the environment database.
        Returns None if no valid parameters exist.
        """
        ...

    @abstractmethod
    def build_task(
        self, params: ScenarioParam, persona: Optional[str] = None
    ) -> Task:
        """Build a complete Task from the sampled parameters."""
        ...

    @abstractmethod
    def verify_task(self, task: Task, env_constructor: Callable[[], Environment]) -> bool:
        """
        Verify the task by running golden actions against a fresh environment.
        Returns True if verification passes.
        """
        ...


class TaskSynthesizer:
    """
    Generates tasks by combining scenario templates with sampled parameters.

    This is the main entry point for task synthesis. It:
    1. Takes a list of ScenarioTemplates
    2. For each template, samples N parameter sets from the database
    3. Builds tasks from each (template, params) pair
    4. Self-verifies each task
    5. Optionally applies persona overlays and difficulty controls
    """

    def __init__(
        self,
        domain: str,
        templates: list[ScenarioTemplate],
        env_constructor: Callable[[], Environment],
        personas: Optional[dict[str, Optional[str]]] = None,
    ):
        self.domain = domain
        self.templates = templates
        self.env_constructor = env_constructor
        self.personas = personas or {"None": None}

    def generate_tasks(
        self,
        num_per_template: int = 5,
        seed: int = 42,
        verify: bool = True,
    ) -> list[Task]:
        """Generate tasks across all templates."""
        rng = random.Random(seed)
        all_tasks = []

        for template in self.templates:
            print(f"Generating tasks for template: {template.name}")
            env = self.env_constructor()
            persona_keys = list(self.personas.keys())
            generated = 0
            attempts = 0
            max_attempts = num_per_template * 10  # Allow retries for sampling

            while generated < num_per_template and attempts < max_attempts:
                attempts += 1
                params = template.sample_params(env, rng)
                if params is None:
                    continue

                persona_key = persona_keys[generated % len(persona_keys)]
                persona_text = self.personas[persona_key]

                try:
                    task = template.build_task(params, persona=persona_text)
                    # Embed persona info in task ID
                    task.id = f"synth_{template.name}_{generated}[PERSONA:{persona_key}]"

                    if verify:
                        if template.verify_task(task, self.env_constructor):
                            all_tasks.append(task)
                            generated += 1
                            print(f"  ✓ Task {task.id} verified")
                        else:
                            print(f"  ✗ Task verification failed, retrying...")
                    else:
                        all_tasks.append(task)
                        generated += 1
                except Exception as e:
                    print(f"  ✗ Error building task: {e}")
                    continue

            print(f"  Generated {generated}/{num_per_template} tasks")

        return all_tasks

    def save_tasks(self, tasks: list[Task], path: str) -> None:
        """Save tasks to JSON file."""
        with open(path, "w") as f:
            json.dump([t.model_dump() for t in tasks], f, indent=2)
        print(f"Saved {len(tasks)} tasks to {path}")
