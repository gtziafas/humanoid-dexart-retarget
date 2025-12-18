import time
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from yourdfpy import URDF
from scipy.spatial.transform import Rotation
import trimesh

from utils import (
    get_mapping_from_mano_to_brainco, augment_object_trajectory, create_laplacian_interaction_mesh
)

class RetargetingWeights(TypedDict):
    joint_smoothness: float         # joint smoothness.
    root_smoothness: float          # root pose smoothness.
    laplacian_deformation: float    # Laplacian deformation energy over interaction mesh (https://arxiv.org/pdf/2509.26633).
    floor_contact: float            # keep the robot's foot in contact with the floor.
    self_collision: float           # self-penetration penalty.
    # priors for maintaining human-like posture.
    posture_prior: float            # keep robot body as close to canonical posture as possible.
    arm_prior: float                # separate weights for shoulders + elbows.
    root_upright: float             # keep torso and head upright.


def main(urdf_path, asset_dir, task_id, obj_augm=False):
    # HOI example data directory conventions (modify accordingly)
    subject_id = "subject1"
    folder = "o1"
    episode_len = -1

    # ----------------------------------------------------------------------
    # Load robot (G1 + Brainco hands)
    # ----------------------------------------------------------------------
    urdf = URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # ----------------------------------------------------------------------
    # Load MANO keypoints (T, 42, 3)
    # ----------------------------------------------------------------------
    asset_path = Path(asset_dir)
    keypoints_path = asset_path / "mano" / subject_id / folder / (task_id + ".mano.npy")
    mano_keypoints = onp.load(keypoints_path)[:episode_len]
    if mano_keypoints.ndim == 2:
        mano_keypoints = mano_keypoints[None, ...]
    assert mano_keypoints.ndim == 3
    num_timesteps, num_mano_joints, _ = mano_keypoints.shape
    assert num_mano_joints == 42, f"Expected 42 MANO joints, got {num_mano_joints}"

    # ----------------------------------------------------------------------
    # Load object mesh + object poses (T, 4, 4)
    # ----------------------------------------------------------------------
    object_mesh_path = "/".join([asset_dir, "object_mesh", task_id, task_id + ".obj"])
    obj_mesh = trimesh.load(object_mesh_path)

    object_pose_path = asset_path / "object_pose" / subject_id / folder / (task_id + ".object.npy")
    object_pose_list = onp.load(object_pose_path)[:episode_len].astype(onp.float32)  # (T,4,4)
    object_poses_se3 = jaxlie.SE3.from_matrix(jnp.array(object_pose_list))

    # ----------------------------------------------------------------------
    # Sample object keypoints in LOCAL mesh coords
    # ----------------------------------------------------------------------
    kps_rnd_idx = onp.random.choice(len(obj_mesh.vertices), size=128, replace=False)
    object_keypoints_local = jnp.array(obj_mesh.vertices[None, kps_rnd_idx].repeat(num_timesteps, 0))

    # ----------------------------------------------------------------------
    # Retargeting index mapping: MANO joints -> G1+Brainco links
    # ----------------------------------------------------------------------
    mano_joint_retarget_indices, g1_joint_retarget_indices = get_mapping_from_mano_to_brainco(robot)

    # ----------------------------------------------------------------------
    # Laplacian source interaction mesh (use corrected object_poses_se3 + corrected mano_keypoints)
    # ----------------------------------------------------------------------
    laplacian_L_source, neighbor_idx, neighbor_mask, neighbor_degree = create_laplacian_interaction_mesh(
        object_keypoints_local,
        object_poses_se3,
        mano_keypoints,
        onp.array(mano_joint_retarget_indices),
    )

    # ----------------------------------------------------------------------
    # Viewer setup
    # ----------------------------------------------------------------------
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    playing = server.gui.add_checkbox("playing", False)
    timestep_slider = server.gui.add_slider("timestep", 0, max(0, num_timesteps - 1), 1, 0)

    object_root = server.scene.add_frame("/object_root", show_axes=False)
    object_handle = server.scene.add_mesh_trimesh("/object_root", obj_mesh)

    grid = server.scene.add_grid("/grid", 2.0, 2.0)
    grid.position = onp.array([0, 0, -0.775])

    # --- Optional debug toggles ---
    show_human_kps = server.gui.add_checkbox("human keypoints", True)
    show_object_kps = server.gui.add_checkbox("object keypoints", True)
    show_object_frame = server.gui.add_checkbox("object frame", False)
    show_world_frame = server.gui.add_checkbox("world frame", False)

    # World frame axes
    world_axes = server.scene.add_frame(
        "/world_axes", 
        show_axes=True,
        axes_radius=0.005,
        axes_length=0.12,
    )
    world_axes.position = onp.array([0.0, 0.0, -0.775])
    world_axes.wxyz = onp.array([1.0, 0.0, 0.0, 0.0])  # identity

    # Object local axes (will be updated each timestep)
    object_axes = server.scene.add_frame(
        "/object_axes", 
        show_axes=True, 
        axes_radius=0.005,  
        axes_length=0.12,   
    )

    # --- Point cloud handles (create once, then update) ---
    # Human keypoints (init at t=0)
    human_pc = server.scene.add_point_cloud(
        "/target_keypoints",
        onp.array(mano_keypoints[0]),
        onp.array((0, 0, 255), dtype=onp.uint8)[None].repeat(num_mano_joints, axis=0),
        point_size=0.005,
    )

    # Object keypoints (init at t=0)
    obj_pose_0 = jax.tree.map(lambda x: x[0], object_poses_se3)
    obj_kps_world_0 = obj_pose_0.apply(object_keypoints_local[0])
    object_pc = server.scene.add_point_cloud(
        "/object_keypoints",
        onp.array(obj_kps_world_0),
        onp.array((255, 0, 0), dtype=onp.uint8)[None].repeat(obj_kps_world_0.shape[0], axis=0),
        point_size=0.005,
    )

    weights = pk.viewer.WeightTuner(
        server,
        RetargetingWeights(  # type: ignore
            laplacian_deformation=1.0,
            joint_smoothness=1.0,
            root_smoothness=50.0,
            posture_prior=5.0,
            root_upright=5.0,
            arm_prior=0.2,
            self_collision=0.5,
            floor_contact=5.0,
        ),
    )

    gen_button = server.gui.add_button("Retarget!")
    #gen_button = weights.gui.add_button("Retarget!")  # <-- NOTE: weights.gui, not server.gui
    gen_button.on_click(lambda _: generate_trajectory())

    # ----------------------------------------------------------------------
    # Object augmentation panel (NEW)
    # ----------------------------------------------------------------------
    # Defaults match your script
    aug_enable = server.gui.add_checkbox("Object Augmentation", obj_augm)

    # Put the controls under a folder/panel if your viser version supports it.
    # If it doesn't, these will still appear in the main GUI list.
    try:
        folder_ctx = server.gui.add_folder("Augmentation Params")
        # collapse by default if supported
        if hasattr(folder_ctx, "is_open"):
            folder_ctx.is_open = False
        elif hasattr(folder_ctx, "open"):
            folder_ctx.open = False
    except Exception:
        folder_ctx = None

    def _add_aug_controls():
        # Translation (meters)
        delta_px = server.gui.add_number("delta_px (m)", 0.2)
        delta_py = server.gui.add_number("delta_py (m)", -0.1)
        delta_pz = server.gui.add_number("delta_pz (m)", 0.0)

        # Rotation (degrees about x,y,z)
        delta_rx = server.gui.add_number("delta_rx (deg)", 0.0)
        delta_ry = server.gui.add_number("delta_ry (deg)", 0.0)
        delta_rz = server.gui.add_number("delta_rz (deg)", 90.0)

        # Timing
        tm_idx = server.gui.add_number("tm_idx (frames)", 20)
        tau_p = server.gui.add_number("tau_p", 60.0)
        tau_theta = server.gui.add_number("tau_theta", 60.0)
        dt = server.gui.add_number("dt", 1.0)

        return delta_px, delta_py, delta_pz, delta_rx, delta_ry, delta_rz, tm_idx, tau_p, tau_theta, dt

    if folder_ctx is not None:
        with folder_ctx:
            (delta_px, delta_py, delta_pz,
             delta_rx, delta_ry, delta_rz,
             tm_idx, tau_p, tau_theta, dt) = _add_aug_controls()
    else:
        (delta_px, delta_py, delta_pz,
         delta_rx, delta_ry, delta_rz,
         tm_idx, tau_p, tau_theta, dt) = _add_aug_controls()

    # ----------------------------------------------------------------------
    # Retargeting state
    # ----------------------------------------------------------------------
    Ts_world_root, joints = None, None
    object_pose_list_aug = object_pose_list
    object_poses_se3_aug = object_poses_se3

    # ----------------------------------------------------------------------
    # Button to run retargeting (UPDATED: reads augmentation UI)
    # ----------------------------------------------------------------------
    def generate_trajectory():
        nonlocal Ts_world_root, joints, object_pose_list_aug, object_poses_se3_aug

        gen_button.disabled = True

        # --- Build augmented object poses (or not), from UI ---
        if aug_enable.value:
            dp = onp.array([float(delta_px.value), float(delta_py.value), float(delta_pz.value)], dtype=onp.float32)
            dr_deg = onp.array([float(delta_rx.value), float(delta_ry.value), float(delta_rz.value)], dtype=onp.float32)
            dr = onp.deg2rad(dr_deg).astype(onp.float32)

            object_pose_list_aug = augment_object_trajectory(
                object_pose_list,
                delta_p=dp,
                delta_rotvec=dr,
                tm_idx=int(tm_idx.value),
                tau_p=float(tau_p.value),
                tau_theta=float(tau_theta.value),
                dt=float(dt.value),
            )
            object_poses_se3_aug = jaxlie.SE3.from_matrix(jnp.array(object_pose_list_aug))
        else:
            object_pose_list_aug = object_pose_list
            object_poses_se3_aug = object_poses_se3

        current_weights = weights.get_weights()  # type: ignore

        Ts_world_root, joints = solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            object_keypoints=object_keypoints_local,
            object_poses=object_poses_se3_aug,   # <-- augmented or not
            g1_joint_retarget_indices_body=g1_joint_retarget_indices,
            weights=current_weights,
            laplacian_L_source=laplacian_L_source,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            neighbor_degree=neighbor_degree,
        )
        gen_button.disabled = False

    # Optional: an "Apply" button specifically for augmentation changes
    apply_aug_button = server.gui.add_button("Apply Augmentation + Retarget!")
    apply_aug_button.on_click(lambda _: generate_trajectory())

    # Also re-run when toggling augmentation checkbox
    aug_enable.on_update(lambda _: generate_trajectory())

    # Run once on start
    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    # ----------------------------------------------------------------------
    # Visualization loop (UPDATED: show/hide + frame toggles + reuse handles)
    # ----------------------------------------------------------------------
    while True:
        with server.atomic():
            if playing.value and num_timesteps > 1:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps

            tstep = int(timestep_slider.value)

            # --- Root pose
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])

            # --- Robot configuration
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            # --- World frame visibility
            world_axes.visible = bool(show_world_frame.value)

            # --- Human keypoints visibility + update
            human_pc.visible = bool(show_human_kps.value)
            if human_pc.visible:
                human_pc.points = onp.array(mano_keypoints[tstep])

            # --- Object pose (use augmented poses for viz)
            t_obj = min(tstep, object_pose_list_aug.shape[0] - 1)
            obj_pose_t = jax.tree.map(lambda x: x[t_obj], object_poses_se3_aug)

            object_root.position = onp.array(obj_pose_t.wxyz_xyz[4:])
            object_root.wxyz = onp.array(obj_pose_t.wxyz_xyz[:4])

            object_handle.position = onp.array(obj_pose_t.wxyz_xyz[4:])
            object_handle.wxyz = onp.array(obj_pose_t.wxyz_xyz[:4])

            # --- Object local frame visibility + update
            object_axes.visible = bool(show_object_frame.value)
            if object_axes.visible:
                object_axes.position = onp.array(obj_pose_t.wxyz_xyz[4:])
                object_axes.wxyz = onp.array(obj_pose_t.wxyz_xyz[:4])

            # --- Object keypoints visibility + update
            object_pc.visible = bool(show_object_kps.value)
            if object_pc.visible:
                obj_kps_world = obj_pose_t.apply(object_keypoints_local[tstep])
                object_pc.points = onp.array(obj_kps_world)

        time.sleep(0.03)

@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    object_keypoints: jnp.ndarray,
    object_poses: jaxlie.SE3,
    g1_joint_retarget_indices_body: jnp.ndarray,        # for Laplacian
    weights: RetargetingWeights,
    laplacian_L_source: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    neighbor_mask: jnp.ndarray,
    neighbor_degree: jnp.ndarray,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """
    Trajectory retargeting with:

       - Laplacian deformation energy on interaction meshes
         (source: human+object, target: robot+object)
       - joint smoothness
       - root smoothness (translation only)
       - joint limits
    """

    timesteps = laplacian_L_source.shape[0]

    # ----------------
    # Robot properties
    # ----------------
    target_ground_z = jnp.array([-0.7568637]) # pelvis-ankle G1 dist in canonical pose.

    dof = len(robot.joints.actuated_names)
    q_canon = jnp.zeros((dof,), dtype=jnp.float32)

    body_joint_names = [
        # Waist / torso
        "waist_yaw_joint",
        "waist_roll_joint",

        # Left leg
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",

        # Right leg
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    body_joint_idx = jnp.array(
        [robot.joints.actuated_names.index(n) for n in body_joint_names if n in robot.joints.actuated_names],
        dtype=jnp.int32,
    )

    # - Foot and head indices.
    left_foot_idx  = robot.links.names.index("left_ankle_roll_link")
    right_foot_idx = robot.links.names.index("right_ankle_roll_link")
    head_idx = robot.links.names.index("head_link")

    # --- Arm joint indices (for arm-only acceleration) ---
    arm_joint_names = [
        # left arm
        "left_shoulder_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        # right arm
        "right_shoulder_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]
    arm_joint_idx = jnp.array(
        [robot.joints.actuated_names.index(n) for n in arm_joint_names if n in robot.joints.actuated_names],
        dtype=jnp.int32,
    )

    # ----------------
    # Variables
    # ----------------
    var_joints = robot.joint_var_cls(jnp.arange(timesteps))  # one joint cfg per timestep
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))  # one SE3 root pose per timestep

    # ----------------
    # Costs
    # ----------------
    num_obj_kp = object_keypoints.shape[1]
    num_total_kp = neighbor_mask.shape[1]
    num_body_kp = num_total_kp - num_obj_kp
    assert num_body_kp == g1_joint_retarget_indices_body.shape[0], (
        "Body portion of interaction mesh must match number of mapped robot joints"
    )

    @jaxls.Cost.create_factory
    def laplacian_deformation_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        obj_pose_t: jaxlie.SE3,
        obj_kps_t: jnp.ndarray,
        L_src_t: jnp.ndarray,           # [N_total, 3]
        neighbor_idx_t: jnp.ndarray,    # [N_total, max_deg]
        neighbor_mask_t: jnp.ndarray,   # [N_total, max_deg]
        neighbor_degree_t: jnp.ndarray, # [N_total, 1]
    ) -> jax.Array:
        """
        Laplacian deformation over body + object.
        Fingers *are* part of this interaction mesh.
        """
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link

        # Robot position -> express in object local frame.
        body_pos = T_world_link.translation()[g1_joint_retarget_indices_body]   # [Nb_body, 3]
        body_pos = obj_pose_t.inverse().apply(body_pos)

        P_t = jnp.concatenate([body_pos, obj_kps_t], axis=0)          # [N_total, 3]

        P_neighbors = P_t[neighbor_idx_t]                                       # [N, max_deg, 3]
        neighbor_sum = (P_neighbors * neighbor_mask_t[..., None]).sum(axis=1)   # [N, 3]
        neighbor_mean = neighbor_sum / neighbor_degree_t                        # [N, 3]

        L_tgt_t = P_t - neighbor_mean                                           # [N, 3]

        residual = (L_tgt_t - L_src_t).reshape(-1)
        return residual * weights["laplacian_deformation"]

    @jaxls.Cost.create_factory
    def floor_contact_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        #var_offset: OffsetVar,
    ) -> jax.Array:
        """Cost to place the robot on the floor:
        - match foot keypoint positions, and
        - penalize the foot from tilting too much.
        """
        T_world_root = var_values[var_Ts_world_root]
        T_root_link = jaxlie.SE3(
            robot.forward_kinematics(cfg=var_values[var_robot_cfg])
        )
        T_world_link = T_world_root @ T_root_link

        left_foot_pos = (
            T_world_link.translation()[left_foot_idx]
        ) 
        right_foot_pos = (
            T_world_link.translation()[right_foot_idx]
        ) 
        left_foot_contact_cost = left_foot_pos[2] - target_ground_z
        right_foot_contact_cost = right_foot_pos[2] - target_ground_z

        # Also penalize the foot from tilting too much -- keep z axis up!
        left_foot_ori = (
           T_world_link.rotation().as_matrix()[left_foot_idx]
        )
        right_foot_ori = (
           T_world_link.rotation().as_matrix()[right_foot_idx]
        )
        left_foot_contact_residual_rot = left_foot_ori[2, 2] - 1
        right_foot_contact_residual_rot = right_foot_ori[2, 2] - 1

        return (
            jnp.concatenate(
                [
                    left_foot_contact_cost.flatten(),
                    right_foot_contact_cost.flatten(),
                    left_foot_contact_residual_rot.flatten(),
                    right_foot_contact_residual_rot.flatten(),
                ]
            )
            * weights["floor_contact"]
        )

    @jaxls.Cost.create_factory
    def root_smoothness_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root_curr: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost for the robot root translation."""
        res_pos = (
            var_values[var_Ts_world_root_curr].translation()
            - var_values[var_Ts_world_root_prev].translation()
        ).flatten()
        return res_pos * weights["root_smoothness"]
        # res_orn = (
        #     var_values[var_Ts_world_root_prev].rotation().inverse() 
        #     @ 
        #     var_values[var_Ts_world_root_curr].rotation()
        # ).log().flatten()
        #return (res_pos + res_orn) * weights["root_smoothness"]

    @jaxls.Cost.create_factory
    def posture_prior_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        q = var_values[var_robot_cfg]
        # Only penalize selected body joints
        res = (q[body_joint_idx] - q_canon[body_joint_idx]).flatten()
        return res * weights["posture_prior"]

    @jaxls.Cost.create_factory
    def arm_prior_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        q = var_values[var_robot_cfg]
        # Only penalize selected arm joints
        res = (q[arm_joint_idx] - q_canon[arm_joint_idx]).flatten()
        return res * weights["arm_prior"]
    
    @jaxls.Cost.create_factory
    def root_upright_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link

        Rw = T_world_root.rotation().as_matrix()
        # Want robot Z axis to align with world Z
        root_residual =  jnp.array([
            Rw[2, 2] - 1.0,
            Rw[0, 2],
            Rw[1, 2],
        ]) 
        
        # Want head Z to also align with world Z
        Rhead = (
            T_world_link.rotation().as_matrix()[head_idx]
        ) 
        head_residual = jnp.array([
            Rhead[2, 2] - 1.0,
            Rhead[0, 2],
            Rhead[1, 2]
        ])

        return (
            jnp.concatenate(
                [
                    root_residual.flatten(),
                    head_residual.flatten(),
                ]
            )
            *  weights["root_upright"]
        )

    costs: list[jaxls.Cost] = [
        laplacian_deformation_cost(
            var_Ts_world_root,
            var_joints,
            object_poses,           # [T] SE3
            object_keypoints,       # [T, N_obj, 3]
            laplacian_L_source,     # [T, N_total, 3]
            neighbor_idx,           # [T, N_total, max_deg]
            neighbor_mask,          # [T, N_total, max_deg]
            neighbor_degree,        # [T, N_total, 1]
        ),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            100.0,
        ),
        posture_prior_cost(var_joints),
        root_upright_cost(var_Ts_world_root, var_joints),
        arm_prior_cost(var_joints),
        # self-penetration
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.05,
            weight=weights['self_collision'],
        ),
        # floor contact
        floor_contact_cost(
            var_Ts_world_root,
            var_joints,
        )
    ]

    # Temporal terms if we have a trajectory
    if timesteps > 1:
        # Joint smoothness
        costs.append(
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                weights["joint_smoothness"],
            )
        )
        # Root smoothness
        costs.append(
            root_smoothness_cost(
                jaxls.SE3Var(jnp.arange(1, timesteps)),
                jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
            )
        )

    # Solve
    problem = jaxls.LeastSquaresProblem(
        costs, [var_joints, var_Ts_world_root]
    ).analyze()

    solution = problem.solve()
    Ts_world_root = solution[var_Ts_world_root]
    return Ts_world_root, solution[var_joints]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="SMPLX → G1+Brainco retargeting with Laplacian deformation."
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        help="Path to robot URDF (e.g., g1_29dof_mode_15_brainco_hand.urdf)",
        default="./assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf"
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        help="Root directory containing MANO, object poses and meshes.",
        default="./example_data/hoi"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task identifier (used as prefix in .smplx.npy and .object.npy filenames).",
    )
    parser.add_argument(
        '--obj_augm',
        action='store_true',
        help='Whether to perform object pose augmentation.'
    )

    args = parser.parse_args()

    main(
        urdf_path=args.urdf_path,
        asset_dir=args.asset_dir,
        task_id=args.task_id,
        obj_augm=args.obj_augm,
    )