import warnings

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, FloatLogSlider, FloatSlider, HBox, Output, Tab, VBox
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", category=DeprecationWarning)

t = np.arange(0, 20, 1e-3)
n = 3


def sub(x):
    return "".join(chr(8272 + ord(c)) for c in str(x + 1))


def simulate(
    intrinsic_freqs,
    intrinsic_amps,
    coupling_weights,
    phase_biases,
    convergence_coefs,
    init_phases,
    init_magnitudes,
):
    n = len(intrinsic_freqs)

    def fun(_, y, w, phi, nu, R, alpha):
        theta, r = y[:n], y[n:]
        dtheta_dt = 2 * np.pi * nu + r @ (
            w * np.sin(np.subtract.outer(theta, theta) - phi)
        )
        dr_dt = alpha * (R - r)
        return np.concatenate([dtheta_dt, dr_dt])

    y0 = np.concatenate([init_phases, init_magnitudes])
    args = (
        coupling_weights,
        phase_biases,
        intrinsic_freqs,
        intrinsic_amps,
        convergence_coefs,
    )
    sol = solve_ivp(fun, [t[0], t[-1]], y0, args=args, t_eval=t)
    theta, r = sol.y[:n].T, sol.y[n:].T
    return theta, r


intrinsic_freqs = np.ones(3)
intrinsic_amps = np.array([1.0, 1.1, 1.2])
coupling_weights = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)
phase_biases = np.deg2rad(
    np.array(
        [
            [0, 120, 0],
            [-120, 0, 120],
            [0, -120, 0],
        ]
    )
)

convergence_coefs = np.ones(3)

rng = np.random.RandomState(0)
init_phases = rng.rand(n) * 2 * np.pi
init_magnitudes = rng.rand(n) * intrinsic_amps


def interactive_plot():
    nu_sliders = [
        FloatLogSlider(
            value=intrinsic_freqs[i],
            base=10,
            min=-2,
            max=2,
            step=0.1,
            description=f"ν{sub(i)}",
        )
        for i in range(n)
    ]
    R_sliders = [
        FloatSlider(value=intrinsic_amps[i], min=0, max=2, description=f"R{sub(i)}")
        for i in range(n)
    ]
    w_sliders = [
        FloatSlider(
            value=coupling_weights[i, j], min=0, max=2, description=f"w{sub(i)}{sub(j)}"
        )
        for i in range(n)
        for j in range(n)
    ]
    phi_sliders = [
        FloatSlider(
            value=np.rad2deg(phase_biases[i, j]),
            min=-180,
            max=180,
            description=f"φ{sub(i)}{sub(j)}",
        )
        for i in range(n)
        for j in range(n)
    ]
    alpha_sliders = [
        FloatSlider(value=convergence_coefs[i], min=0, max=2, description=f"α{sub(i)}")
        for i in range(n)
    ]
    theta0_slider = [
        FloatSlider(
            value=np.rad2deg(init_phases[i]),
            min=0,
            max=360,
            description=f"θ{sub(i)}(0)",
        )
        for i in range(n)
    ]
    r0_slider = [
        FloatSlider(
            value=init_magnitudes[i],
            min=0,
            max=1.5,
            description=f"r{sub(i)}(0)",
            step=0.01,
        )
        for i in range(n)
    ]

    tabs = {
        "intrinsic_freqs": nu_sliders,
        "intrinsic_amps": R_sliders,
        "coupling_weights": w_sliders,
        "phase_biases": phi_sliders,
        "convergence_coefs": alpha_sliders,
        "init_phases": theta0_slider,
        "init_magnitudes": r0_slider,
    }

    for key, sliders in tabs.items():
        if len(sliders) == 9:
            tabs[key] = VBox(
                [HBox([sliders[i * 3 + j] for j in range(3)]) for i in range(3)]
            )
        else:
            tabs[key] = VBox(sliders)

    tab = Tab()
    tab.children = list(tabs.values())
    tab.titles = list(tabs.keys())

    output = Output()

    with output:
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, tight_layout=True)

    theta, r = simulate(
        intrinsic_freqs,
        intrinsic_amps,
        coupling_weights,
        phase_biases,
        convergence_coefs,
        init_phases,
        init_magnitudes,
    )
    theta_lines = axs[0].plot(t, theta % (2 * np.pi), linewidth=1)
    axs[0].set_yticks([0, np.pi, 2 * np.pi])
    axs[0].set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    axs[0].set_ylabel("Phase")
    r_lines = axs[1].plot(t, r, linewidth=1)
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(
        min(slider.min for slider in r0_slider), max(slider.max for slider in r0_slider)
    )

    def update(*args):
        intrinsic_freqs = np.array([slider.value for slider in nu_sliders])
        intrinsic_amps = np.array([slider.value for slider in R_sliders])
        coupling_weights = np.array([slider.value for slider in w_sliders]).reshape(
            n, n
        )
        phase_biases = np.deg2rad(
            np.array([slider.value for slider in phi_sliders])
        ).reshape(n, n)
        convergence_coefs = np.array([slider.value for slider in alpha_sliders])
        init_phases = np.deg2rad(np.array([slider.value for slider in theta0_slider]))
        init_magnitudes = np.array([slider.value for slider in r0_slider])
        theta, r = simulate(
            intrinsic_freqs,
            intrinsic_amps,
            coupling_weights,
            phase_biases,
            convergence_coefs,
            init_phases,
            init_magnitudes,
        )
        for i, line in enumerate(theta_lines):
            line.set_ydata(theta.T[i] % (2 * np.pi))
        for i, line in enumerate(r_lines):
            line.set_ydata(r.T[i])
        fig.canvas.draw_idle()

    def reset(*args):
        for nu, slider in zip(intrinsic_freqs, nu_sliders):
            slider.value = nu

        for R, slider in zip(intrinsic_amps, R_sliders):
            slider.value = R

        for w, slider in zip(coupling_weights.ravel(), w_sliders):
            slider.value = w

        for phi, slider in zip(phase_biases.ravel(), phi_sliders):
            slider.value = np.rad2deg(phi)

        for alpha, slider in zip(convergence_coefs, alpha_sliders):
            slider.value = alpha

        for theta, slider in zip(init_phases, theta0_slider):
            slider.value = np.rad2deg(theta)

        for R, slider in zip(init_magnitudes, r0_slider):
            slider.value = R

    for slider in (
        nu_sliders
        + R_sliders
        + w_sliders
        + phi_sliders
        + alpha_sliders
        + theta0_slider
        + r0_slider
    ):
        slider.observe(update, "value")

    button = Button(description="Reset")
    button.on_click(reset)
    return VBox([tab, button, output])
