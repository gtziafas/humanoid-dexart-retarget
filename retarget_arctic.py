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

from utils import (
    get_mapping_from_smplx_to_g1_29_brainco, SMPLX_JOINT_NAMES, augment_object_trajectory, create_laplacian_interaction_mesh
)

class RetargetingWeights(TypedDict):
    joint_smoothness: float      # joint smoothness.
    root_smoothness: float       # root translation smoothness.
    laplacian_deformation: float # Laplacian deformation energy over interaction mesh (https://arxiv.org/pdf/2509.26633).
    floor_contact: float         # keep the robot's foot in contact with the floor.
    self_collision: float        # self-penetration penalty.

def main(urdf_path, asset_dir, task_id, obj_augm=False):
    """
    Retarget a SMPLX human trajectory (T x J x 3) to Unitree G1 with Brainco hands,
    using only:

      - Laplacian deformation energy of an interaction mesh
      - Joint smoothness
      - Root smoothness (optional)
      - Joint limit penalties
      - Floor contact regularization

    No explicit global/local keypoint alignment.
    """
    # ARCTIC example data directory conventions (modify accordingly)
    subject_id = "s04"
    object_id = task_id.split('_')[0]

    # ----------------------------------------------------------------------
    # Load robot (G1 + Brainco hands)
    # ----------------------------------------------------------------------
    urdf = URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # ----------------------------------------------------------------------
    # Load SMPLX keypoints (scaled offline to roughly match G1 height, and aligned with MANO)
    #   Expect shape (T, J, 3). J must match len(SMPLX_JOINT_NAMES).
    # ----------------------------------------------------------------------
    asset_path = Path(asset_dir)
    keypoints_path = asset_path / "smplx_aligned" / subject_id / (task_id + ".smplx.npy")
    smplx_keypoints = onp.load(keypoints_path)  # [T, J, 3]

    if smplx_keypoints.ndim == 2:  # (J, 3) -> (1, J, 3)
        smplx_keypoints = smplx_keypoints[None, ...]
    assert smplx_keypoints.ndim == 3
    num_timesteps, num_smplx_joints, _ = smplx_keypoints.shape
    assert num_smplx_joints == len(SMPLX_JOINT_NAMES), (
        f"Expected {len(SMPLX_JOINT_NAMES)} SMPLX joints, "
        f"got {num_smplx_joints} in keypoints."
    )

    # ----------------------------------------------------------------------
    # Load object URDF + mesh + poses
    # ----------------------------------------------------------------------
    object_urdf_path = "/".join([asset_dir, "object_urdf", object_id, object_id + ".urdf"])
    obj_urdf = URDF.load(object_urdf_path)

    # Object pose trajectory (same world frame as SMPLX)
    object_pose_path = asset_path / "object_pose" / object_id / subject_id.replace("0", "") / (task_id + ".object.npy")
    object_pose_raw = onp.load(object_pose_path)  # [T, 7] (articulation, rx, ry, rz, tx, ty, tz)

    object_articulation = object_pose_raw[:, 0]                # [T]
    object_pose_trans_list = object_pose_raw[:, 4:7] / 1000.0  # mm -> m
    object_pose_rot_list = Rotation.from_rotvec(object_pose_raw[:, 1:4]).as_matrix()
    object_pose_list = onp.eye(4)[None].repeat(num_timesteps, 0)
    object_pose_list[:, :3, :3] = object_pose_rot_list
    object_pose_list[:, :3, 3]  = object_pose_trans_list
    del object_pose_trans_list, object_pose_rot_list, object_pose_raw

    # JAX SE3 for Laplacian cost (shape [T])
    object_poses_se3 = jaxlie.SE3.from_matrix(jnp.array(object_pose_list))

    # Rollout articulation + pose over time to save consistent sampled object keypoints correctly.
    kps_rnd_idx = onp.random.choice(len(obj_urdf.scene.to_geometry().vertices), size=128, replace=False)
    object_keypoints_local = []
    for t in range(num_timesteps):
        obj_urdf.update_cfg(onp.array([object_articulation[t]]))
        mesh = obj_urdf.scene.to_geometry()
        object_keypoints_local.append(jnp.array(mesh.vertices[kps_rnd_idx]))
    object_keypoints_local = jnp.stack(object_keypoints_local)
    
    # ----------------------------------------------------------------------
    # Retargeting index mapping: SMPLX joints -> G1+Brainco links
    # ----------------------------------------------------------------------
    (
        smplx_joint_retarget_indices,
        g1_joint_retarget_indices,
    ) = get_mapping_from_smplx_to_g1_29_brainco()

    smplx_joint_retarget_indices_np = onp.array(
        smplx_joint_retarget_indices, dtype=onp.int32
    )
    g1_joint_retarget_indices_np = onp.array(
        g1_joint_retarget_indices, dtype=onp.int32
    )

    # Laplacian source interaction mesh
    laplacian_L_source, neighbor_idx, neighbor_mask, neighbor_degree = create_laplacian_interaction_mesh(
        object_keypoints_local, object_poses_se3, smplx_keypoints, smplx_joint_retarget_indices_np
    )

    # ----------------------------------------------------------------------
    # Viewer setup (UPDATED: toggles + frames + augmentation panel)
    # ----------------------------------------------------------------------
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    playing = server.gui.add_checkbox("playing", False)
    timestep_slider = server.gui.add_slider("timestep", 0, max(0, num_timesteps - 1), 1, 0)

    object_root = server.scene.add_frame("/object_root", show_axes=False)
    object_vis = ViserUrdf(server, obj_urdf, root_node_name="/object_root")

    grid = server.scene.add_grid("/grid", 2.0, 2.0)
    grid.position = onp.array([0, 0, 0.125])

    # ----------------------------------------------------------------------
    # Debug toggles (NEW)
    # ----------------------------------------------------------------------
    show_human_kps = server.gui.add_checkbox("human keypoints", True)
    show_object_kps = server.gui.add_checkbox("object keypoints", True)
    show_object_frame = server.gui.add_checkbox("object frame", False)
    show_world_frame = server.gui.add_checkbox("world frame", False)

    # World frame axes (create once, toggle visibility)
    world_axes = server.scene.add_frame(
        "/world_axes", 
        show_axes=True,
        axes_radius=0.005,
        axes_length=0.12,
    )
    world_axes.position = onp.array([0.0, 0.0, 0.0])
    world_axes.wxyz = onp.array([1.0, 0.0, 0.0, 0.0])  # identity

    # Object local axes (create once, updated per timestep)
    object_axes = server.scene.add_frame(
        "/object_axes", 
        show_axes=True, 
        axes_radius=0.005,  
        axes_length=0.12,   
    )

    # Point clouds (create once, update per timestep)
    human_pc = server.scene.add_point_cloud(
        "/target_keypoints",
        onp.array(smplx_keypoints[0]),
        onp.array((0, 0, 255), dtype=onp.uint8)[None].repeat(num_smplx_joints, axis=0),
        point_size=0.005,
    )

    obj_pose_0 = jax.tree.map(lambda x: x[0], object_poses_se3)
    obj_kps_world_0 = obj_pose_0.apply(object_keypoints_local[0])
    object_pc = server.scene.add_point_cloud(
        "/object_keypoints",
        onp.array(obj_kps_world_0),
        onp.array((255, 0, 0), dtype=onp.uint8)[None].repeat(obj_kps_world_0.shape[0], axis=0),
        point_size=0.005,
    )
    
    # ---- Weights panel (unchanged) ----
    weights = pk.viewer.WeightTuner(
        server,
        RetargetingWeights(  # type: ignore
            laplacian_deformation=1.0,
            joint_smoothness=1.0,
            root_smoothness=1.0,
            self_collision=0.5,
            floor_contact=5.0,
        ),
    )

    Ts_world_root, joints = None, None

    # ----------------------------------------------------------------------
    # Retargeting (UPDATED: will use augmented or non-augmented object poses)
    # ----------------------------------------------------------------------
    object_pose_list_aug = object_pose_list
    object_poses_se3_aug = object_poses_se3

    def generate_trajectory():
        nonlocal Ts_world_root, joints, object_pose_list_aug, object_poses_se3_aug

        gen_button.disabled = True

        # build augmented object trajectory from UI state
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
            object_keypoints=object_keypoints_local,           # [T, N_obj, 3]
            object_poses=object_poses_se3_aug,                 # [T] SE3 (aug or not)
            g1_joint_retarget_indices_body=g1_joint_retarget_indices,
            weights=current_weights,
            laplacian_L_source=laplacian_L_source,         # [T, N_total, 3]
            neighbor_idx=neighbor_idx,                     # [T, N_total, max_deg]
            neighbor_mask=neighbor_mask,                   # [T, N_total, max_deg]
            neighbor_degree=neighbor_degree,               # [T, N_total, 1]
        )
        gen_button.disabled = False

    gen_button = server.gui.add_button("Retarget!")
    gen_button.on_click(lambda _: generate_trajectory())

    # ----------------------------------------------------------------------
    # Object augmentation panel (NEW; collapsed if supported)
    # ----------------------------------------------------------------------
    aug_enable = server.gui.add_checkbox("Object Augmentation", obj_augm)

    try:
        aug_folder = server.gui.add_folder("Augmentation Params")
        # collapse by default if supported
        if hasattr(aug_folder, "is_open"):
            aug_folder.is_open = False
        elif hasattr(aug_folder, "open"):
            aug_folder.open = False
    except Exception:
        aug_folder = None

    def _add_aug_controls():
        # Translation (meters)
        _delta_px = server.gui.add_number("delta_px (m)", 0.2)
        _delta_py = server.gui.add_number("delta_py (m)", -0.1)
        _delta_pz = server.gui.add_number("delta_pz (m)", 0.0)

        # Rotation (degrees about x,y,z)
        _delta_rx = server.gui.add_number("delta_rx (deg)", 0.0)
        _delta_ry = server.gui.add_number("delta_ry (deg)", 0.0)
        _delta_rz = server.gui.add_number("delta_rz (deg)", 90.0)

        # Timing
        _tm_idx = server.gui.add_number("tm_idx (frames)", 20)
        _tau_p = server.gui.add_number("tau_p", 60.0)
        _tau_theta = server.gui.add_number("tau_theta", 60.0)
        _dt = server.gui.add_number("dt", 1.0)
        return _delta_px, _delta_py, _delta_pz, _delta_rx, _delta_ry, _delta_rz, _tm_idx, _tau_p, _tau_theta, _dt

    if aug_folder is not None:
        with aug_folder:
            (delta_px, delta_py, delta_pz,
             delta_rx, delta_ry, delta_rz,
             tm_idx, tau_p, tau_theta, dt) = _add_aug_controls()

            apply_aug_button = server.gui.add_button("Apply Augmentation + Retarget")
            apply_aug_button.on_click(lambda _: generate_trajectory())
    else:
        (delta_px, delta_py, delta_pz,
         delta_rx, delta_ry, delta_rz,
         tm_idx, tau_p, tau_theta, dt) = _add_aug_controls()

        apply_aug_button = server.gui.add_button("Apply Augmentation + Retarget")
        apply_aug_button.on_click(lambda _: generate_trajectory())

    # Also re-run when toggling augmentation checkbox (optional convenience)
    aug_enable.on_update(lambda _: generate_trajectory())

    # Run once on start (now uses GUI augmentation state)
    generate_trajectory()
    assert Ts_world_root is not None and joints is not None

    # ----------------------------------------------------------------------
    # Visualization loop (UPDATED: show/hide + reuse handles + obj frame)
    # ----------------------------------------------------------------------
    while True:
        with server.atomic():
            if playing.value and num_timesteps > 1:
                timestep_slider.value = (timestep_slider.value + 1) % num_timesteps

            tstep = int(timestep_slider.value)

            # Root pose.
            base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
            base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])

            # Robot configuration.
            urdf_vis.update_cfg(onp.array(joints[tstep]))

            # World frame toggle
            world_axes.visible = bool(show_world_frame.value)

            # Human keypoints toggle + update (reuse handle)
            human_pc.visible = bool(show_human_kps.value)
            if human_pc.visible:
                human_pc.points = onp.array(smplx_keypoints[tstep])

            # Object pose (augmented or not)
            t_obj = min(tstep, object_pose_list_aug.shape[0] - 1)
            obj_pose_t = jax.tree.map(lambda x: x[t_obj], object_poses_se3_aug)

            object_root.position = onp.array(obj_pose_t.wxyz_xyz[4:])
            object_root.wxyz = onp.array(obj_pose_t.wxyz_xyz[:4])

            # articulation DOF
            a = float(object_articulation[t_obj])
            object_vis.update_cfg(onp.array([a], dtype=onp.float32))

            # Object local frame toggle + update
            object_axes.visible = bool(show_object_frame.value)
            if object_axes.visible:
                object_axes.position = onp.array(obj_pose_t.wxyz_xyz[4:])
                object_axes.wxyz = onp.array(obj_pose_t.wxyz_xyz[:4])

            # Object keypoints toggle + update (reuse handle)
            object_pc.visible = bool(show_object_kps.value)
            if object_pc.visible:
                object_keypoints_world = obj_pose_t.apply(object_keypoints_local[tstep])
                object_pc.points = onp.array(object_keypoints_world)

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

    # - Foot indices.
    left_foot_idx  = robot.links.names.index("left_ankle_roll_link")
    right_foot_idx = robot.links.names.index("right_ankle_roll_link")
    target_ground_z = jnp.array([0.147361]) # average human-g1 height dist.

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
        res = (
            var_values[var_Ts_world_root_curr].translation()
            - var_values[var_Ts_world_root_prev].translation()
        ).flatten()
        return res * weights["root_smoothness"]

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
        ),
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
        help="Root directory containing SMPLX, object poses, URDFs, and templates.",
        default="./example_data/arctic"
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