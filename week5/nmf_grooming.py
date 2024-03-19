from flygym.mujoco import NeuroMechFly
from flygym.mujoco.state import KinematicPose
from flygym.mujoco.arena.tethered import Tethered

import matplotlib.pyplot as plt

import pickle
from pathlib import Path
import numpy as np

# List of all the joints that are actuated during grooming
all_groom_dofs = (
    [f"joint_{dof}" for dof in ["Head", "Head_yaw", "Head_roll"]]  # Head joints
    + [
        f"joint_{side}F{dof}"
        for side in "LR"
        for dof in [
            "Coxa",
            "Coxa_roll",
            "Coxa_yaw",
            "Femur",
            "Femur_roll",
            "Tibia",
            "Tarsus1",
        ]
    ]  # Front leg joints
    + [
        f"joint_{side}{dof}{angle}"
        for side in "LR"
        for dof in ["Pedicel"]
        for angle in ["", "_yaw"]
    ]  # Antennae joints
)

# List of alL the bodies that might be colliding during groomming

groom_self_collision = [
    f"{side}{app}"
    for side in "LR"
    for app in [
        "FTibia",
        "FTarsus1",
        "FTarsus2",
        "FTarsus3",
        "FTarsus4",
        "FTarsus5",
        "Arista",
        "Funiculus",
        "Pedicel",
        "Eye",
    ]
]


class NeuromechflyGrooming(NeuroMechFly):
    # Bas class for grooming set the initial pose, potential collisions and actuated joints
    # Makes it possible to add touch sensors to monitor the contacts of the antennas with the legs

    def __init__(
        self,
        sim_params=None,
        actuated_joints=all_groom_dofs,
        arena=None,
        xml_variant="deepfly3d_old",
        groom_collision=False,
        touch_sensor_locations=[],
    ):
        self.touch_sensor_locations = touch_sensor_locations
        collisions = []
        if groom_collision:
            collisions = groom_self_collision

        if arena is None:
            arena = Tethered()
        super().__init__(
            sim_params=sim_params,
            actuated_joints=actuated_joints,
            arena=arena,
            xml_variant=xml_variant,
            self_collisions=collisions,
            floor_collisions="none",
            init_pose=KinematicPose.from_yaml("./data/pose_groom.yaml"),
        )
        self._zoom_camera()

    def _set_joints_stiffness_and_damping(self):
        super()._set_joints_stiffness_and_damping()
        # set the stiffness and damping of antennal joints
        for joint in self.model.find_all("joint"):
            if any([app in joint.name for app in ["Pedicel", "Arista", "Funiculus"]]):
                joint.stiffness = 1e-3
                joint.damping = 1e-3

        return None

    def _set_actuators_gain(self):
        for actuator in self._actuators:
            if "Arista" in actuator.name:
                kp = 1e-6
            elif "Pedicel" in actuator.name or "Funiculus" in actuator.name:
                kp = 0.2
            else:
                kp = 20.0
            actuator.kp = kp
        return None

    def _zoom_camera(self):
        if self.sim_params.render_camera == "Animat/camera_front":
            self.physics.model.camera(self.sim_params.render_camera).pos0 = [
                2.7,
                0.0,
                1.5,
            ]
        elif self.sim_params.render_camera == "Animat/camera_right":
            self.physics.model.camera(self.sim_params.render_camera).pos0 = [
                0.65,
                -2.0,
                1.25,
            ]
        elif self.sim_params.render_camera == "Animat/camera_left":
            self.physics.model.camera(self.sim_params.render_camera).pos0 = [
                0.65,
                2.0,
                1.25,
            ]

    def _define_self_contacts(self, self_collisions_geoms):
        # Only add relevant collisions:
        # - No collisions between the two antennas
        # - No collisions between segments in the same leg or antenna

        self_contact_pairs = []
        self_contact_pairs_names = []

        for geom1_name in self_collisions_geoms:
            for geom2_name in self_collisions_geoms:
                body1 = self.model.find("geom", geom1_name).parent
                body2 = self.model.find("geom", geom2_name).parent
                simple_body1_name = body1.name.split("_")[0]
                simple_body2_name = body2.name.split("_")[0]

                body1_children = self.get_real_childrens(body1)
                body2_children = self.get_real_childrens(body2)

                body1_parent = self._get_real_parent(body1)
                body2_parent = self._get_real_parent(body2)

                geom1_is_antenna = any(
                    [
                        app in geom1_name
                        for app in ["Pedicel", "Arista", "Funiculus", "Eye"]
                    ]
                )
                geom2_is_antenna = any(
                    [
                        app in geom2_name
                        for app in ["Pedicel", "Arista", "Funiculus", "Eye"]
                    ]
                )
                is_same_side = geom1_name[0] == geom2_name[0]

                if not (
                    body1.name == body2.name
                    or simple_body1_name in body2_children
                    or simple_body2_name in body1_children
                    or simple_body1_name == body2_parent
                    or simple_body2_name == body1_parent
                    or geom1_is_antenna
                    and geom2_is_antenna  # both on antenna
                    or is_same_side
                    and (
                        not geom1_is_antenna and not geom2_is_antenna
                    )  # on the legs and same side
                ):
                    contact_pair = self.model.contact.add(
                        "pair",
                        name=f"{geom1_name}_{geom2_name}",
                        geom1=geom1_name,
                        geom2=geom2_name,
                        solref=self.sim_params.contact_solref,
                        solimp=self.sim_params.contact_solimp,
                        margin=0.0,  # change margin to avoid penetration
                    )
                    self_contact_pairs.append(contact_pair)
                    self_contact_pairs_names.append(f"{geom1_name}_{geom2_name}")

        return self_contact_pairs, self_contact_pairs_names

    def _add_force_sensors(self):
        super()._add_force_sensors()
        self._add_touch_sensors()

    def _add_touch_sensors(self):
        touch_sensors = []
        for body_name in self.touch_sensor_locations:
            body = self.model.find("body", body_name)
            body_child_names = self.get_real_childrens(body)
            if body_child_names:
                body_child_name = body_child_names[0]
                body_child = self.model.find("body", body_child_name)
                next_body_pos = body_child.pos
                if np.sum(np.abs(next_body_pos)) < 1e-3:
                    next_body_pos = [0.0, 0.0, -0.2]
            elif "Arista" in body_name:
                if body_name[0] == "L":
                    next_body_pos = [0.0, -0.2, 0.0]
                else:
                    next_body_pos = [0.0, 0.2, 0.0]
            elif "Funiculus" in body_name:
                next_body_pos = [0.0, 0.0, -0.2]
            elif "Tarsus" in body_name:
                next_body_pos = [0.0, 0.0, -0.2]
            elif "Eye" in body_name:
                pass
            else:
                raise ValueError(f"Body {body_name} has no children")
            if "Eye" in body_name:
                site = body.add(
                    "site",
                    type="sphere",
                    name=f"{body_name}_touchsite",
                    pos=next_body_pos,
                    size=[1.0, 0.0, 0.0],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                )
            else:
                quat = body.quat
                site = body.add(
                    "site",
                    type="capsule",
                    quat=quat,
                    fromto=np.hstack((np.zeros(3), next_body_pos)),
                    name=f"{body_name}_touchsite",
                    pos=[0, 0, 0],
                    size=[0.1, 0.0, 0.0],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                )
            touch_sensor = self.model.sensor.add(
                "touch", name=f"touch_{body.name}", site=site.name
            )
            touch_sensors.append(touch_sensor)
        self.touch_sensors = touch_sensors

    def get_observation(self):
        obs = super().get_observation()
        touch_data = self.physics.bind(self.touch_sensors).sensordata
        obs["touch_sensors"] = touch_data.copy()
        return obs


def load_grooming_data(data_path, timestep):
    # Loads the full grooming data adapting the timestep and the joint order

    data_path = Path(data_path)
    # load the data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    run_time = len(data["joint_LFCoxa"]) * data["meta"]["timestep"]
    target_num_steps = int(run_time / timestep)
    data_block = np.zeros((len(all_groom_dofs), target_num_steps))
    input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    output_t = np.arange(target_num_steps) * timestep
    for i, joint in enumerate(all_groom_dofs):
        if "Pedicel" in joint:
            pass
            # data[joint] = np.ones_like(data[joint])*np.mean(data[joint])
        if "RPedicel_yaw" in joint:
            # data[joint] = np.ones_like(data[joint])*np.mean(data[joint])*-1
            data[joint] *= -1
        data_block[i, :] = np.unwrap(np.interp(output_t, input_t, data[joint]))

    # swap head roll and head yaw
    hroll_id = all_groom_dofs.index("joint_Head_roll")
    hyaw_id = all_groom_dofs.index("joint_Head_yaw")
    hroll = data_block[hroll_id, :].copy()
    data_block[hroll_id, :] = data_block[hyaw_id, :]
    data_block[hyaw_id, :] = hroll * 2.0

    return data_block


modules_colors = {"R_antenna": "r", "L_antenna": "b", "foreleg": "g", "eyes": "y"}

appendage_bodies = {
    "antenna": ["Pedicel", "Funiculus", "Arista"],
    "foreleg": ["Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5", "Tibia"],
    "eye": ["Eye"],
}


def plot_state_and_contacts(
    time,
    touch_sensor_data,
    touch_sensor_locations,
    states,
    transit_times,
    dust_levels=None,
    dusted_appendages=None,
    n_cols=2,
    plot_appendages=["Rantenna", "Lantenna", "Rforeleg", "Lforeleg", "Reye", "Leye"],
):
    # plot touch sensor traces as well as behavior of the animal and the dust levels if provided

    n_cols = 2
    n_rows = len(plot_appendages) // n_cols
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 4 * n_cols), sharey=True
    )
    axs = axs.flatten()

    touch_sensors_belongings = []

    for appendage in plot_appendages:
        touch_sensor_belongs = []
        for i, touch_sensor in enumerate(touch_sensor_locations):
            same_side = appendage[0] == touch_sensor[0]
            isin_appendage = any(
                [
                    appendage_body in touch_sensor
                    for appendage_body in appendage_bodies[appendage[1:]]
                ]
            )
            if same_side and isin_appendage:
                touch_sensor_belongs.append(i)
        touch_sensors_belongings.append(touch_sensor_belongs)

    for i, ax in enumerate(axs):
        ax.plot(
            time,
            np.sum(touch_sensor_data[:, touch_sensors_belongings[i]], axis=1),
            color="k",
        )
        ax.set_title(plot_appendages[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (mN)")
        ax.set_xlim(time[0] - 0.1 * time[-1], time[-1] + 0.1 * time[-1])
        prev_module_start = transit_times[0]
        assert prev_module_start == 0
        prev_module = states[0]
        legend_dict = {}
        for next_module, next_module_start in zip(states[1:], transit_times[1:]):
            lines = ax.axvspan(
                time[prev_module_start],
                time[next_module_start],
                color=modules_colors[prev_module],
                alpha=0.1,
            )
            if prev_module not in legend_dict:
                legend_dict[prev_module] = lines
            prev_module_start = next_module_start + 1
            prev_module = next_module
        lines = ax.axvspan(
            time[prev_module_start],
            time[-1],
            color=modules_colors[prev_module],
            alpha=0.1,
        )
        if prev_module not in legend_dict:
            legend_dict[prev_module] = lines
        ax.legend(
            legend_dict.values(), legend_dict.keys(), loc="center left", fontsize=8
        )
        if dust_levels is not None:
            twin_ax = ax.twinx()
            lines = twin_ax.plot(time, dust_levels, ls="--")
            twin_ax.set_ylabel("Dust level")
            twin_ax.set_ylim(-0.1, np.max(dust_levels) + 0.1)
            twin_ax.legend(lines, dusted_appendages, loc="center right", fontsize=8)
    plt.tight_layout()
    return fig, axs
