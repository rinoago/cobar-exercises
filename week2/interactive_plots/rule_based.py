import warnings

import matplotlib.pyplot as plt
import numpy as np
from flygym.mujoco.examples.rule_based_controller import RuleBasedSteppingCoordinator
from ipywidgets import Button, FloatSlider, HBox, Output, VBox

warnings.filterwarnings("ignore", category=DeprecationWarning)


legs = ("LF", "LM", "LH", "RF", "RM", "RH")
attrs = (
    "leg_phases",
    "combined_scores",
    "rule1_scores",
    "rule2_scores",
    "rule3_scores",
)
titles = (
    "Leg phases",
    "Stepping scores\n(combined)",
    "Stepping scores\n(rule 1 contribution)",
    "Stepping scores\n(rule 2 contribution)",
    "Stepping scores\n(rule 3 contribution)",
)
spacings = (10, 18, 18, 18, 18)


def simulate(weights, run_time, timestep, rules_graph, preprogrammed_steps):
    controller = RuleBasedSteppingCoordinator(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
    )
    n_steps = int(run_time / timestep)
    data = np.zeros((n_steps, 5, 6))
    for i in range(int(run_time / controller.timestep)):
        controller.step()
        data[i] = [getattr(controller, attr) for attr in attrs]
    return data


def interactive_plot(
    run_time,
    timestep,
    rules_graph,
    default_weights,
    preprogrammed_steps,
):
    sliders = [
        FloatSlider(weight, min=-15, max=15, description=rule, continuous_update=False)
        for rule, weight in default_weights.items()
    ]

    reset_button = Button(description="Reset")

    controls = VBox([HBox(sliders[:3]), HBox(sliders[3:] + [reset_button])])

    n_steps = int(run_time / timestep)
    t = np.arange(n_steps) * timestep
    keys = list(default_weights)
    offsets = np.outer(spacings, -np.arange(6))

    output = Output()

    with output:
        fig, axs = plt.subplots(5, 1, figsize=(9, 8), tight_layout=True, sharex=True)

    lines = []

    data = simulate(
        default_weights, run_time, timestep, rules_graph, preprogrammed_steps
    )
    data += offsets

    for i, ax in enumerate(axs):
        lines.extend(ax.plot(t, data[:, i]))
        ax.set_yticks(offsets[i], legs)
        ax.set_ylabel(titles[i], rotation=0, ha="right", va="center")
        for offset in offsets[i]:
            ax.axhline(offset, color="k", lw=0.5)

    axs[-1].set_xlabel("Time (s)")

    def update(*args):
        weights = {rule: slider.value for rule, slider in zip(keys, sliders)}
        data = simulate(weights, run_time, timestep, rules_graph, preprogrammed_steps)
        data += offsets
        for line, y in zip(lines, data.transpose(1, 2, 0).reshape((30, -1))):
            line.set_ydata(y)

        fig.canvas.draw_idle()

    def reset(*args):
        for slider in sliders:
            slider.value = default_weights[slider.description]

    for slider in sliders:
        slider.observe(update, "value")

    reset_button.on_click(reset)

    return VBox([output, controls])
