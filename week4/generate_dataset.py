import numpy as np
from tqdm import trange
from flygym.mujoco.core import NeuroMechFly, Parameters
from flygym.mujoco.examples.obstacle_arena import ObstacleOdorArena
from flygym.mujoco.examples.turning_controller import HybridTurningNMF
from flygym.mujoco.arena import FlatTerrain


class ArenaWithFly2(ObstacleOdorArena):
    def __init__(self, height=0.2, **kwargs):
        super().__init__(
            obstacle_positions=np.array([[0, 0]]),
            obstacle_colors=np.array([[0, 0, 0, 0]]) / 255,
            terrain=FlatTerrain(),
            obstacle_height=1e-4,
            **kwargs,
        )

        self.height = height
        fly = NeuroMechFly().model
        fly.detach()
        fly.model = "Animat_2"

        for light in fly.find_all(namespace="light"):
            light.remove()

        spawn_site = self.root_element.worldbody.add(
            "site",
            pos=(0, 0, self.height),
            euler=(0, 0, 0),
        )
        self.freejoint = spawn_site.attach(fly).add("freejoint")

    def set_fly2_position(self, physics, obs, r, theta, phi):
        fly1_heading = obs["fly_orientation"][:2] @ (1, 1j)
        fly1_heading /= np.abs(fly1_heading)
        fly1_pos = obs["fly"][0, :2] @ (1, 1j)

        fly2_pos = r * np.exp(1j * theta) * fly1_heading + fly1_pos
        fly2_heading = np.exp(1j * phi) * fly1_heading

        q = np.exp(1j * np.angle(fly2_heading) / 2)
        qpos = (fly2_pos.real, fly2_pos.imag, self.height, q.real, 0, 0, q.imag)
        physics.bind(self.freejoint).qpos = qpos


from utils import crop_hex_to_rect

contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

arena = ArenaWithFly2()

sim_params = Parameters(
    render_playspeed=1e-4,
    render_camera="Animat/camera_top_zoomout",
    enable_vision=True,
    render_raw_vision=True,
    enable_olfaction=True,
    vision_refresh_rate=10000,
    render_fps=1,
    render_window_size=(256, 256),
)

nmf = HybridTurningNMF(
    sim_params=sim_params,
    arena=arena,
    spawn_pos=(0, 0, 0.2),
    spawn_orientation=(0, 0, 0),
    contact_sensor_placements=contact_sensor_placements,
)

obs = nmf.reset(seed=0)[0]
images = []

n_steps = 10000
rng = np.random.RandomState(0)

r = rng.uniform(1.5, 10, n_steps)
theta = rng.uniform(-np.pi, np.pi, n_steps)
phi = rng.uniform(-np.pi, np.pi, n_steps)

for i in trange(500):
    nmf.step(np.array([1, 1]))

for i in trange(n_steps):
    obs = nmf.step(np.array([1, 1]))[0]
    arena.set_fly2_position(nmf.physics, obs, r=r[i], theta=theta[i], phi=phi[i])

    nmf._last_vision_update_time = -np.inf  # hack: this forces visual input update
    nmf._update_vision()
    images.append(crop_hex_to_rect(nmf._curr_visual_input.copy()))

    nmf.render()
images = np.array(images, dtype=np.float32)
r = r.astype(np.float32)
theta = theta.astype(np.float32)
np.savez_compressed("data/data.npz", images=images, r=r, theta=theta)
