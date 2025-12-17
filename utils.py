import pyroki as pk
import jax
import jax.numpy as jnp
import numpy as onp
from scipy.spatial.transform  import Rotation
from scipy.spatial import Delaunay

def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Create a NxN connectivity matrix for N links.
    The matrix is marked Y if there is a direct kinematic chain connection
    between the two links, without bypassing the root link.
    """
    n = len(link_indices)
    conn_matrix = jnp.zeros((n, n))

    def is_direct_chain_connection(idx1: int, idx2: int) -> bool:
        """Check if two joints are connected in the kinematic chain without other retargeted joints between"""
        joint1 = link_indices[idx1]
        joint2 = link_indices[idx2]

        # Check path from joint2 up to root
        current = joint2
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint1:
                return True
            if parent in link_indices:
                # Hit another retargeted joint before finding joint1
                break
            current = parent

        # Check path from joint1 up to root
        current = joint1
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint2:
                return True
            if parent in link_indices:
                # Hit another retargeted joint before finding joint2
                break
            current = parent

        return False

    # Build symmetric connectivity matrix
    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)  # Self-connection
        for j in range(i + 1, n):
            if is_direct_chain_connection(i, j):
                conn_matrix = conn_matrix.at[i, j].set(1.0)
                conn_matrix = conn_matrix.at[j, i].set(1.0)

    return conn_matrix


def create_laplacian_interaction_mesh(object_keypoints, object_poses_se3, smplx_keypoints, smplx_joint_retarget_indices):
    # ----------------------------------------------------------------------
    # Laplacian interaction mesh precomputation (source motion)
    #
    # Interaction mesh at each timestep t:
    #   - body keypoints: SMPLX joints used for Laplacian.
    #   - object keypoints: sampled vertices properly articulated.
    #
    # We compute Delaunay per-frame and build per-frame neighbor lists.
    # Laplacian coordinates are then precomputed for the source
    # (human + object) trajectory.
    # ----------------------------------------------------------------------
    num_timesteps = smplx_keypoints.shape[0]
    num_body_kp = smplx_joint_retarget_indices.shape[0]
    num_obj_kp = object_keypoints.shape[1]
    num_total_kp = num_body_kp + num_obj_kp

    # First pass: compute neighbors per frame, track global max degree
    neighbors_all = []
    max_deg_global = 0

    for t in range(num_timesteps):
        # Body points at this timestep
        smplx_t = smplx_keypoints[t, smplx_joint_retarget_indices, :]  # [Nb_body, 3]

        # Object points at this timestem -- local frame
        obj_t_local = object_keypoints[t]

        # Construct at object local frame
        obj_pose_t = jax.tree.map(lambda x: x[t], object_poses_se3)
        smplx_t_local = obj_pose_t.inverse().apply(smplx_t)

        P_t = onp.concatenate([smplx_t_local, obj_t_local], axis=0)   # [N_total, 3]

        # Delaunay per-frame
        delaunay = Delaunay(P_t)
        tetra = delaunay.simplices  # [M, 4]

        neighbors_t = [[] for _ in range(num_total_kp)]
        for tet in tetra:
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = int(tet[i]), int(tet[j])
                    neighbors_t[a].append(b)
                    neighbors_t[b].append(a)

        neighbors_t = [sorted(set(nbs)) for nbs in neighbors_t]
        deg_t = max(len(nbs) for nbs in neighbors_t)
        max_deg_global = max(max_deg_global, deg_t)
        neighbors_all.append(neighbors_t)

    if max_deg_global == 0:
        raise ValueError("All vertices have degree 0 in the interaction mesh.")

    neighbor_idx_np  = onp.zeros((num_timesteps, num_total_kp, max_deg_global), dtype=onp.int32)
    neighbor_mask_np = onp.zeros((num_timesteps, num_total_kp, max_deg_global), dtype=onp.float32)
    degree_np        = onp.zeros((num_timesteps, num_total_kp, 1),              dtype=onp.float32)

    for t in range(num_timesteps):
        neighbors_t = neighbors_all[t]
        for i, nbs in enumerate(neighbors_t):
            d = len(nbs)
            if d == 0:
                continue
            neighbor_idx_np[t, i, :d] = onp.array(nbs, dtype=onp.int32)
            neighbor_mask_np[t, i, :d] = 1.0
            degree_np[t, i, 0] = float(d)

    # Avoid division by zero
    degree_np[degree_np == 0.0] = 1.0

    laplacian_L_source = onp.zeros((num_timesteps, num_total_kp, 3), dtype=onp.float32)

    for t in range(num_timesteps):
        # body at t
        smplx_t = smplx_keypoints[t, smplx_joint_retarget_indices, :]  # [Nb_body, 3]

        # object at t
        obj_t_local = object_keypoints[t]                       # [No, 3]

        # Construct at object local frame
        obj_pose_t = jax.tree.map(lambda x: x[t], object_poses_se3)
        smplx_t_local = obj_pose_t.inverse().apply(smplx_t)
        # obj_t_local = obj_pose_t.inverse().apply(obj_t)

        P_t = onp.concatenate([smplx_t_local, obj_t_local], axis=0)             # [N_total, 3]

        P_neighbors = P_t[neighbor_idx_np[t]]                                   # [N, max_deg, 3]
        neighbor_sum = (P_neighbors * neighbor_mask_np[t][..., None]).sum(axis=1)  # [N, 3]
        neighbor_mean = neighbor_sum / degree_np[t]                             # [N, 3]
        laplacian_L_source[t] = P_t - neighbor_mean                             # [N, 3]

    # Convert to JAX arrays to feed into the solver
    laplacian_L_source = jnp.array(laplacian_L_source)  # [T, N_total, 3]
    neighbor_idx = jnp.array(neighbor_idx_np)           # [T, N_total, max_deg]
    neighbor_mask = jnp.array(neighbor_mask_np)         # [T, N_total, max_deg]
    neighbor_degree = jnp.array(degree_np)              # [T, N_total, 1]

    return laplacian_L_source, neighbor_idx, neighbor_mask, neighbor_degree



SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

SMPLX_JOINT_NAMES = [
 'pelvis',
 'left_hip',
 'right_hip',
 'spine1',
 'left_knee',
 'right_knee',
 'spine2',
 'left_ankle',
 'right_ankle',
 'spine3',
 'left_foot',
 'right_foot',
 'neck',
 'left_collar',
 'right_collar',
 'head',
 'left_shoulder',
 'right_shoulder',
 'left_elbow',
 'right_elbow',
 'left_wrist',
 'right_wrist',
 'jaw',
 'left_eye_smplhf',
 'right_eye_smplhf',
 'left_index1',
 'left_index2',
 'left_index3',
 'left_middle1',
 'left_middle2',
 'left_middle3',
 'left_pinky1',
 'left_pinky2',
 'left_pinky3',
 'left_ring1',
 'left_ring2',
 'left_ring3',
 'left_thumb1',
 'left_thumb2',
 'left_thumb3',
 'right_index1',
 'right_index2',
 'right_index3',
 'right_middle1',
 'right_middle2',
 'right_middle3',
 'right_pinky1',
 'right_pinky2',
 'right_pinky3',
 'right_ring1',
 'right_ring2',
 'right_ring3',
 'right_thumb1',
 'right_thumb2',
 'right_thumb3',
 'nose',
 'right_eye',
 'left_eye',
 'right_ear',
 'left_ear',
 'left_big_toe',
 'left_small_toe',
 'left_heel',
 'right_big_toe',
 'right_small_toe',
 'right_heel',
 'left_thumb',
 'left_index',
 'left_middle',
 'left_ring',
 'left_pinky',
 'right_thumb',
 'right_index',
 'right_middle',
 'right_ring',
 'right_pinky',
 'right_eye_brow1',
 'right_eye_brow2',
 'right_eye_brow3',
 'right_eye_brow4',
 'right_eye_brow5',
 'left_eye_brow5',
 'left_eye_brow4',
 'left_eye_brow3',
 'left_eye_brow2',
 'left_eye_brow1',
 'nose1',
 'nose2',
 'nose3',
 'nose4',
 'right_nose_2',
 'right_nose_1',
 'nose_middle',
 'left_nose_1',
 'left_nose_2',
 'right_eye1',
 'right_eye2',
 'right_eye3',
 'right_eye4',
 'right_eye5',
 'right_eye6',
 'left_eye4',
 'left_eye3',
 'left_eye2',
 'left_eye1',
 'left_eye6',
 'left_eye5',
 'right_mouth_1',
 'right_mouth_2',
 'right_mouth_3',
 'mouth_top',
 'left_mouth_3',
 'left_mouth_2',
 'left_mouth_1',
 'left_mouth_5',
 'left_mouth_4',
 'mouth_bottom',
 'right_mouth_4',
 'right_mouth_5',
 'right_lip_1',
 'right_lip_2',
 'lip_top',
 'left_lip_2',
 'left_lip_1',
 'left_lip_3',
 'lip_bottom',
 'right_lip_3'
]

# When loaded from `g1_description`s 23-dof model.
G1_LINK_NAMES = [
    "pelvis",
    "pelvis_contour_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_roll_link",
    "logo_link",
    "imu_link",
    "left_palm_link",
    "left_zero_link",
    "left_one_link",
    "left_two_link",
    "left_three_link",
    "left_four_link",
    "left_five_link",
    "left_six_link",
    "right_palm_link",
    "right_zero_link",
    "right_one_link",
    "right_two_link",
    "right_three_link",
    "right_four_link",
    "right_five_link",
    "right_six_link",
]

# When loaded from `g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf` model of `unitree_ros/robots`.
G1_29_BRAINCO_LINK_NAMES = [
 'pelvis',
 'pelvis_contour_link',
 'left_hip_pitch_link',
 'left_hip_roll_link',
 'left_hip_yaw_link',
 'left_knee_link',
 'left_ankle_pitch_link',
 'left_ankle_roll_link',
 'right_hip_pitch_link',
 'right_hip_roll_link',
 'right_hip_yaw_link',
 'right_knee_link',
 'right_ankle_pitch_link',
 'right_ankle_roll_link',
 'waist_yaw_link',
 'waist_roll_link',
 'torso_link',
 'logo_link',
 'head_link',
 'imu_in_torso',
 'imu_in_pelvis',
 'd435_link',
 'mid360_link',
 'left_shoulder_pitch_link',
 'left_shoulder_roll_link',
 'left_shoulder_yaw_link',
 'left_elbow_link',
 'left_wrist_roll_link',
 'left_wrist_pitch_link',
 'left_wrist_yaw_link',
 'left_base2_link',
 'left_base_link',
 'left_thumb_metacarpal_Link',
 'left_thumb_proximal_Link',
 'left_thumb_distal_Link',
 'left_thumb_tip_Link',
 'left_index_proximal_Link',
 'left_index_distal_Link',
 'left_index_tip_Link',
 'left_middle_proximal_Link',
 'left_middle_distal_Link',
 'left_middle_tip_Link',
 'left_ring_proximal_Link',
 'left_ring_distal_Link',
 'left_ring_tip_Link',
 'left_pinky_proximal_Link',
 'left_pinky_distal_Link',
 'left_pinky_tip_Link',
 'right_shoulder_pitch_link',
 'right_shoulder_roll_link',
 'right_shoulder_yaw_link',
 'right_elbow_link',
 'right_wrist_roll_link',
 'right_wrist_pitch_link',
 'right_wrist_yaw_link',
 'right_base2_link',
 'right_base_link',
 'right_thumb_metacarpal_link',
 'right_thumb_proximal_link',
 'right_thumb_distal_link',
 'right_thumb_tip',
 'right_index_proximal_link',
 'right_index_distal_link',
 'right_index_tip',
 'right_middle_proximal_link',
 'right_middle_distal_link',
 'right_middle_tip',
 'right_ring_proximal_link',
 'right_ring_distal_link',
 'right_ring_tip',
 'right_pinky_proximal_link',
 'right_pinky_distal_link',
 'right_pinky_tip'
]

MANO_TO_SHADOW_MAPPING = {
    # Wrist
    0: "palm",
    # Thumb
    1: "thhub",
    2: "thmiddle",
    3: "thdistal",
    4: "thtip",
    # Index
    5: "ffproximal",
    6: "ffmiddle",
    7: "ffdistal",
    8: "fftip",
    # Middle
    9: "mfproximal",
    10: "mfmiddle",
    11: "mfdistal",
    12: "mftip",
    # Ring
    13: "rfproximal",
    14: "rfmiddle",
    15: "rfdistal",
    16: "rftip",
    # # Little
    17: "lfproximal",
    18: "lfmiddle",
    19: "lfdistal",
    20: "lftip",
}

MANO_TO_BRAINCO_MAPPING = {
    'left': {
        # Wrist
        0: f"left_base_link",
        # Thumb
        1: "left_thumb_metacarpal_Link",
        2: "left_thumb_proximal_Link",
        3: "left_thumb_distal_Link",
        4: "left_thumb_tip_Link",
        # Index
        5: "left_index_proximal_Link",
        7: "left_index_distal_Link",
        8: "left_index_tip_Link",
        # Middle
        9: "left_middle_proximal_Link",
        11: "left_middle_distal_Link",
        12: "left_middle_tip_Link",
        # Ring
        13: "left_ring_proximal_Link",
        15: "left_ring_distal_Link",
        16: "left_ring_tip_Link",
        # # Little
        17: "left_pinky_proximal_Link",
        19: "left_pinky_distal_Link",
        20: "left_pinky_tip_Link",
    },
    'right': {
        # Wrist
        0: f"right_base_link",
        # Thumb
        1: "right_thumb_metacarpal_link",
        2: "right_thumb_proximal_link",
        3: "right_thumb_distal_link",
        4: "right_thumb_tip",
        # Index
        5: "right_index_proximal_link",
        7: "right_index_distal_link",
        8: "right_index_tip",
        # Middle
        9: "right_middle_proximal_link",
        11: "right_middle_distal_link",
        12: "right_middle_tip",
        # Ring
        13: "right_ring_proximal_link",
        15: "right_ring_distal_link",
        16: "right_ring_tip",
        # # Little
        17: "right_pinky_proximal_link",
        19: "right_pinky_distal_link",
        20: "right_pinky_tip",
    },
}

MANO_TO_SMPLX_MAPPING = {
    'left': {
        0: 'left_wrist',
        1: 'left_thumb1',
        2: 'left_thumb2',
        3: 'left_thumb3',
        4: 'left_thumb',
        5: 'left_index1',
        6: 'left_index2',
        7: 'left_index3',
        8: 'left_index',
        9: 'left_middle1',
        10: 'left_middle2',
        11: 'left_middle3',
        12: 'left_middle',
        13: 'left_ring1',
        14: 'left_ring2',
        15: 'left_ring3',
        16: 'left_ring',
        17: 'left_pinky1',
        18: 'left_pinky2',
        19: 'left_pinky3',
        20: 'left_pinky',
    },
    'right': {
        0: 'right_wrist',
        1: 'right_thumb1',
        2: 'right_thumb2',
        3: 'right_thumb3',
        4: 'right_thumb',
        5: 'right_index1',
        6: 'right_index2',
        7: 'right_index3',
        8: 'right_index',
        9: 'right_middle1',
        10: 'right_middle2',
        11: 'right_middle3',
        12: 'right_middle',
        13: 'right_ring1',
        14: 'right_ring2',
        15: 'right_ring3',
        16: 'right_ring',
        17: 'right_pinky1',
        18: 'right_pinky2',
        19: 'right_pinky3',
        20: 'right_pinky',
    }
}

### Dictionaries from human joints to robot joints

def get_mapping_from_smpl_to_g1() -> tuple[jnp.ndarray, jnp.ndarray]:
    smpl_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for smpl_name, g1_name in [
        ("pelvis", "pelvis_contour_link"),
        ("left_hip", "left_hip_pitch_link"),
        ("right_hip", "right_hip_pitch_link"),
        ("left_knee", "left_knee_link"),
        ("right_knee", "right_knee_link"),
        ("left_ankle", "left_ankle_roll_link"),
        ("right_ankle", "right_ankle_roll_link"),
        ("left_shoulder", "left_shoulder_roll_link"),
        ("right_shoulder", "right_shoulder_roll_link"),
        ("left_elbow", "left_elbow_pitch_link"),
        ("right_elbow", "right_elbow_pitch_link"),
        ("left_wrist", "left_palm_link"),
        ("right_wrist", "right_palm_link"),
    ]:
        smpl_joint_retarget_indices_to_g1.append(SMPL_JOINT_NAMES.index(smpl_name))
        g1_joint_retarget_indices.append(G1_LINK_NAMES.index(g1_name))

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return smpl_joint_retarget_indices, g1_joint_retarget_indices

def get_mapping_from_smplx_to_g1_29_brainco() -> tuple[jnp.ndarray, jnp.ndarray]:
    smplx_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for mhr_name, g1_name in [
        # upper body
        ("pelvis", "pelvis_contour_link"),
        ('right_hip', "right_hip_roll_link"),
        ('left_hip', "left_hip_roll_link"),
        ('right_knee', "right_knee_link"),
        ("left_knee", "left_knee_link"),
        ("left_shoulder", "left_shoulder_roll_link"),
        ("right_shoulder", "right_shoulder_roll_link"),
        ("left_elbow", "left_elbow_link"),
        ("right_elbow", "right_elbow_link"),
        ('left_ankle', 'left_ankle_roll_link'),
        ('right_ankle', "right_ankle_roll_link"),
        ('nose_middle', 'mid360_link'),
        # hands
        ('left_wrist', 'left_base_link'),
        ('left_thumb1', 'left_thumb_metacarpal_Link'),
        ('left_thumb2', 'left_thumb_proximal_Link'),
        ('left_thumb3', 'left_thumb_distal_Link'),
        ('left_thumb', 'left_thumb_tip_Link'),
        ('left_index2', 'left_index_proximal_Link'),
        ('left_index3', 'left_index_distal_Link'),
        ('left_index', 'left_index_tip_Link'),
        ('left_middle2', 'left_middle_proximal_Link'),
        ('left_middle3', 'left_middle_distal_Link'),
        ('left_middle', 'left_middle_tip_Link'),
        ('left_ring2', 'left_ring_proximal_Link'),
        ('left_ring3', 'left_ring_distal_Link'),
        ('left_ring', 'left_ring_tip_Link'),
        ('left_pinky2', 'left_pinky_proximal_Link'),
        ('left_pinky3', 'left_pinky_distal_Link'),
        ('left_pinky', 'left_pinky_tip_Link'),
        ('right_wrist', 'right_base_link'),
        ('right_thumb1', 'right_thumb_metacarpal_link'),
        ('right_thumb2', 'right_thumb_proximal_link'),
        ('right_thumb3', 'right_thumb_distal_link'),
        ('right_thumb', 'right_thumb_tip'),
        ('right_index2', 'right_index_proximal_link'),
        ('right_index3', 'right_index_distal_link'),
        ('right_index', 'right_index_tip'),
        ('right_middle2', 'right_middle_proximal_link'),
        ('right_middle3', 'right_middle_distal_link'),
        ('right_middle', 'right_middle_tip'),
        ('right_ring2', 'right_ring_proximal_link'),
        ('right_ring3', 'right_ring_distal_link'),
        ('right_ring', 'right_ring_tip'),
        ('right_pinky2', 'right_pinky_proximal_link'),
        ('right_pinky3', 'right_pinky_distal_link'),
        ('right_pinky', 'right_pinky_tip'),
    ]:
        smplx_joint_retarget_indices_to_g1.append(SMPLX_JOINT_NAMES.index(mhr_name))
        g1_joint_retarget_indices.append(G1_29_BRAINCO_LINK_NAMES.index(g1_name))

    mhr_joint_retarget_indices = jnp.array(smplx_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return mhr_joint_retarget_indices, g1_joint_retarget_indices

def get_mapping_from_mano_to_shadow(robot: pk.Robot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the mapping indices between MANO and Shadow Hand joints."""
    SHADOW_TO_MANO_MAPPING = {v: k for k, v in MANO_TO_SHADOW_MAPPING.items()}
    shadow_joint_idx = []
    mano_joint_idx = []
    link_names = robot.links.names
    for i, link_name in enumerate(link_names):
        if link_name in SHADOW_TO_MANO_MAPPING:
            shadow_joint_idx.append(i)
            mano_joint_idx.append(SHADOW_TO_MANO_MAPPING[link_name])

    return jnp.array(shadow_joint_idx), jnp.array(mano_joint_idx)

def get_mapping_from_mano_to_g1_29_brainco() -> tuple[jnp.ndarray, jnp.ndarray]:
    mano_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for i, (_, g1_name) in enumerate([
        ('left_wrist', 'left_base_link'),
        ('left_thumb1', 'left_thumb_metacarpal_Link'),
        ('left_thumb2', 'left_thumb_proximal_Link'),
        ('left_thumb3', 'left_thumb_distal_Link'),
        ('left_thumb', 'left_thumb_tip_Link'),
        ('left_index2', 'left_index_proximal_Link'),
        ('left_index3', 'left_index_distal_Link'),
        ('left_index', 'left_index_tip_Link'),
        ('left_middle2', 'left_middle_proximal_Link'),
        ('left_middle3', 'left_middle_distal_Link'),
        ('left_middle', 'left_middle_tip_Link'),
        ('left_ring2', 'left_ring_proximal_Link'),
        ('left_ring3', 'left_ring_distal_Link'),
        ('left_ring', 'left_ring_tip_Link'),
        ('left_pinky2', 'left_pinky_proximal_Link'),
        ('left_pinky3', 'left_pinky_distal_Link'),
        ('left_pinky', 'left_pinky_tip_Link'),
        ('right_wrist', 'right_base_link'),
        ('right_thumb1', 'right_thumb_metacarpal_link'),
        ('right_thumb2', 'right_thumb_proximal_link'),
        ('right_thumb3', 'right_thumb_distal_link'),
        ('right_thumb', 'right_thumb_tip'),
        ('right_index2', 'right_index_proximal_link'),
        ('right_index3', 'right_index_distal_link'),
        ('right_index', 'right_index_tip'),
        ('right_middle2', 'right_middle_proximal_link'),
        ('right_middle3', 'right_middle_distal_link'),
        ('right_middle', 'right_middle_tip'),
        ('right_ring2', 'right_ring_proximal_link'),
        ('right_ring3', 'right_ring_distal_link'),
        ('right_ring', 'right_ring_tip'),
        ('right_pinky2', 'right_pinky_proximal_link'),
        ('right_pinky3', 'right_pinky_distal_link'),
        ('right_pinky', 'right_pinky_tip'),
    ]):
        mano_joint_retarget_indices_to_g1.append(i)
        g1_joint_retarget_indices.append(G1_29_BRAINCO_LINK_NAMES.index(g1_name))

    mhr_joint_retarget_indices = jnp.array(mano_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return mhr_joint_retarget_indices, g1_joint_retarget_indices

def get_mapping_from_mhr_to_g1_29_brainco() -> tuple[jnp.ndarray, jnp.ndarray]:
    mhr_joint_retarget_indices_to_g1 = []
    g1_joint_retarget_indices = []

    for mhr_name, g1_name in [
        # upper body
        ("root", "pelvis_contour_link"),
        ('r_upleg', "right_hip_roll_link"),
        ('l_upleg', "left_hip_roll_link"),
        ('r_lowleg', "right_knee_link"),
        ("l_lowleg", "left_knee_link"),
        ("l_uparm", "left_shoulder_roll_link"),
        ("r_uparm", "right_shoulder_roll_link"),
        ("l_lowarm", "left_elbow_link"),
        ("r_lowarm", "right_elbow_link"),
        ('l_foot', 'left_ankle_roll_link'),
        ('r_foot', "right_ankle_roll_link"),
        # hands
        ('l_wrist', 'left_base_link'),
        ('l_thumb0', 'left_thumb_metacarpal_Link'),
        ('l_thumb1', 'left_thumb_proximal_Link'),
        ('l_thumb2', 'left_thumb_distal_Link'),
        ('l_index1', 'left_index_proximal_Link'),
        ('l_index2', 'left_index_distal_Link'),
        ('l_index3', 'left_index_tip_Link'),
        ('l_middle1', 'left_middle_proximal_Link'),
        ('l_middle2', 'left_middle_distal_Link'),
        ('l_middle3', 'left_middle_tip_Link'),
        ('l_ring1', 'left_ring_proximal_Link'),
        ('l_ring2', 'left_ring_distal_Link'),
        ('l_ring3', 'left_ring_tip_Link'),
        ('l_pinky1', 'left_pinky_proximal_Link'),
        ('l_pinky2', 'left_pinky_distal_Link'),
        ('l_pinky3', 'left_pinky_tip_Link'),
        ('r_wrist', 'right_base_link'),
        ('r_thumb0', 'right_thumb_metacarpal_link'),
        ('r_thumb1', 'right_thumb_proximal_link'),
        ('r_thumb2', 'right_thumb_distal_link'),
        ('r_index1', 'right_index_proximal_link'),
        ('r_index2', 'right_index_distal_link'),
        ('r_index3', 'right_index_tip'),
        ('r_middle1', 'right_middle_proximal_link'),
        ('r_middle2', 'right_middle_distal_link'),
        ('r_middle3', 'right_middle_tip'),
        ('r_ring1', 'right_ring_proximal_link'),
        ('r_ring2', 'right_ring_distal_link'),
        ('r_ring3', 'right_ring_tip'),
        ('r_pinky1', 'right_pinky_proximal_link'),
        ('r_pinky2', 'right_pinky_distal_link'),
        ('r_pinky3', 'right_pinky_tip'),
    ]:
        mhr_joint_retarget_indices_to_g1.append(MHR_JOINT_NAMES.index(mhr_name))
        g1_joint_retarget_indices.append(G1_29_BRAINCO_LINK_NAMES.index(g1_name))

    mhr_joint_retarget_indices = jnp.array(mhr_joint_retarget_indices_to_g1)
    g1_joint_retarget_indices = jnp.array(g1_joint_retarget_indices)
    return mhr_joint_retarget_indices, g1_joint_retarget_indices

def get_mapping_from_mano_to_brainco(robot: pk.Robot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the mapping indices between MANO and BRAINCO joints."""
    BRAINCO_TO_MANO_MAPPING = {v: k for k, v in MANO_TO_SHADOW_MAPPING.items()}
    shadow_joint_idx = []
    mano_joint_idx = []
    link_names = robot.links.names
    for i, link_name in enumerate(link_names):
        if link_name in BRAINCO_TO_MANO_MAPPING:
            shadow_joint_idx.append(i)
            mano_joint_idx.append(BRAINCO_TO_MANO_MAPPING[link_name])

    return jnp.array(shadow_joint_idx), jnp.array(mano_joint_idx)

def get_mapping_from_smplx_to_mano() -> tuple[jnp.ndarray, jnp.ndarray]:
    # MANO: (42,), first left then right
    pairs = [
        (20, 0),
        (37, 1),
        (38, 2),
        (39, 3),
        (66, 4),
        (25, 5),
        (26, 6),
        (27, 7),
        (67, 8),
        (28, 9),
        (29, 10),
        (30, 11),
        (68, 12),
        (34, 13),
        (35, 14),
        (36, 15),
        (69, 16),
        (31, 17),
        (32, 18),
        (33, 19),
        (70, 20),
        (21, 21),
        (52, 22),
        (53, 23),
        (54, 24),
        (71, 25),
        (40, 26),
        (41, 27),
        (42, 28),
        (72, 29),
        (43, 30),
        (44, 31),
        (45, 32),
        (73, 33),
        (49, 34),
        (50, 35),
        (51, 36),
        (74, 37),
        (46, 38),
        (47, 39),
        (48, 40),
        (75, 41)
    ]
    mano_joint_index, smplx_joint_index = zip(*pairs)
    return jnp.array(smplx_joint_index), jnp.array(mano_joint_index)


### Perform object pose augmentation based on https://arxiv.org/pdf/2509.26633

def augment_object_trajectory(
    object_pose_list: onp.ndarray, # [T, 4, 4], original
    delta_p: onp.ndarray,          # [3], Δp_obj in meters
    delta_rotvec: onp.ndarray,     # [3], axis-angle Δθ_obj (rad)
    tm_idx: int,                   # onset index t_m
    tau_p: float,                  # decay constant for translation (in timesteps or seconds)
    tau_theta: float,              # decay for rotation
    dt: float = 1.0,               # timestep size (1.0 if you use frame index directly)
) -> onp.ndarray:
    """
    Implements Eq. (14a, 14b) from the paper for a discrete-time trajectory.
    """
    T = object_pose_list.shape[0]
    p = object_pose_list[:, :3, 3]       # [T, 3]
    R = object_pose_list[:, :3, :3]      # [T, 3, 3]

    R_delta_full = Rotation.from_rotvec(delta_rotvec)

    p_aug = onp.zeros_like(p)
    R_aug = onp.zeros_like(R)

    for t in range(T):
        if t < tm_idx:
            # Before object motion: displaced from the *initial* pose
            p_aug[t] = p[0] + delta_p
            R_aug[t] = (R_delta_full * Rotation.from_matrix(R[0])).as_matrix()
        else:
            # Exponential decay of the offset
            # If you treat tau_p, tau_theta in *seconds*, use exp(- (t - tm_idx)*dt / tau)
            # If they are in "frames", set dt=1 and just exp(- (t - tm_idx) / tau).
            decay_p = onp.exp(- (t - tm_idx) * dt / tau_p) if tau_p > 0 else 0.0
            decay_theta = onp.exp(- (t - tm_idx) * dt / tau_theta) if tau_theta > 0 else 0.0

            p_aug[t] = p[t] + decay_p * delta_p

            # Scale the rotation offset in axis-angle space
            R_delta_t = Rotation.from_rotvec(decay_theta * delta_rotvec)
            R_aug[t] = (R_delta_t * Rotation.from_matrix(R[t])).as_matrix()

    object_pose_list_aug = object_pose_list.copy()
    object_pose_list_aug[:, :3, :3] = R_aug
    object_pose_list_aug[:, :3, 3] = p_aug
    return object_pose_list_aug