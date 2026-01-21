#!/usr/bin/env python3
import copy
import os
import sys
import xml.etree.ElementTree as ET

# -----------------------
# USER SETTINGS
# -----------------------
G1_URDF_IN = "g1_29dof_rev_1_0.urdf"
HANDS_URDF_IN = "inspire_hands_raw.urdf"   # wrapper <robot> file you create

G1_URDF_OUT = "g1_29dof_rev_1_0_with_inspire_hands.urdf"

# These exist in the Unitree G1 URDF you mentioned (from unitree_ros config)
G1_REMOVE_LINKS = {"left_rubber_hand", "right_rubber_hand"}
G1_REMOVE_JOINTS = {"left_hand_palm_joint", "right_hand_palm_joint"}

# Mount points on G1
LEFT_PARENT_LINK = "left_wrist_yaw_link"
RIGHT_PARENT_LINK = "right_wrist_yaw_link"

# In your pasted hand snippet, the mount uses left_tool0/right_tool0 as parent.
# We’ll rewrite those to the above.
LEFT_TOOL0_NAME = "left_tool0"
RIGHT_TOOL0_NAME = "right_tool0"

# Mesh rewrite:
# your snippet uses: meshes/visual/... and meshes/collision/...
# but your assets live in: inspire_hands/meshes/visual/... etc.
HAND_ASSET_ROOT = "inspire_hands"  # relative path from the merged URDF

# If your visual meshes are GLB rather than OBJ, set this True.
# It will rewrite ONLY visual meshes *.obj -> *.glb (collision stays as-is).
USE_GLB_FOR_VISUAL = True

# Prefix all LEFT hand links/joints with "l_" to avoid collisions.
LEFT_PREFIX = "l_"

# Right-hand in your snippet already uses r_ for most links/joints, but the mount joint is "rgripper_mount"
# We keep right side as-is.
# -----------------------


def get_name(el):
    return el.attrib.get("name")


def remove_by_name(parent, tag, names_to_remove):
    """Remove children with a given tag and name in names_to_remove."""
    to_remove = []
    for child in list(parent):
        if child.tag != tag:
            continue
        nm = get_name(child)
        if nm in names_to_remove:
            to_remove.append(child)
    for child in to_remove:
        parent.remove(child)


def rewrite_mesh_paths(robot_root, asset_root, use_glb_for_visual):
    """
    Rewrite mesh filename attributes from:
      meshes/visual/foo.obj -> inspire_hands/meshes/visual/foo.(obj|glb)
      meshes/collision/foo.obj -> inspire_hands/meshes/collision/foo.obj
    """
    for mesh in robot_root.findall(".//mesh"):
        fn = mesh.attrib.get("filename")
        if not fn:
            continue

        # Normalize Windows backslashes just in case
        fn_norm = fn.replace("\\", "/")

        if fn_norm.startswith("meshes/"):
            new_fn = f"{asset_root}/{fn_norm}"
            # Optionally swap visual OBJ->GLB
            if use_glb_for_visual and "/meshes/visual/" in new_fn and new_fn.lower().endswith(".obj"):
                new_fn = new_fn[:-4] + ".glb"
            mesh.attrib["filename"] = new_fn


def prefix_names(robot_root, prefix, *, skip_prefix_for=None):
    """
    Add prefix to all:
      - <link name="...">
      - <joint name="...">
      - joint parent/child link references: <parent link="...">, <child link="...">
      - mimic joint references: <mimic joint="...">
    """
    if skip_prefix_for is None:
        skip_prefix_for = set()

    def should_skip(name):
        return (name is None) or (name in skip_prefix_for) or name.startswith(prefix)

    # rename links
    for link in robot_root.findall("./link"):
        nm = link.attrib.get("name")
        if nm and not should_skip(nm):
            link.attrib["name"] = prefix + nm

    # rename joints
    for joint in robot_root.findall("./joint"):
        nm = joint.attrib.get("name")
        if nm and not should_skip(nm):
            joint.attrib["name"] = prefix + nm

    # fix parent/child link refs inside joints
    for joint in robot_root.findall("./joint"):
        p = joint.find("parent")
        c = joint.find("child")
        if p is not None:
            lk = p.attrib.get("link")
            if lk and not should_skip(lk):
                p.attrib["link"] = prefix + lk
        if c is not None:
            lk = c.attrib.get("link")
            if lk and not should_skip(lk):
                c.attrib["link"] = prefix + lk

        mimic = joint.find("mimic")
        if mimic is not None:
            mj = mimic.attrib.get("joint")
            if mj and not should_skip(mj):
                mimic.attrib["joint"] = prefix + mj


def find_joint(robot_root, joint_name):
    for j in robot_root.findall("./joint"):
        if j.attrib.get("name") == joint_name:
            return j
    return None


def retarget_mount_joint_parent(robot_root, mount_joint_name, new_parent_link):
    """
    Find the mount joint and rewrite its <parent link="...">.
    """
    j = find_joint(robot_root, mount_joint_name)
    if j is None:
        raise RuntimeError(f"Could not find mount joint '{mount_joint_name}' in hands URDF")

    parent_el = j.find("parent")
    if parent_el is None:
        parent_el = ET.SubElement(j, "parent")
    parent_el.attrib["link"] = new_parent_link


def extract_left_right_subtrees(hands_root):
    """
    Your file contains both left and right hand elements mixed in one <robot>.
    We’ll split them into two Element lists:
      left_elems: elements belonging to left hand (based on names)
      right_elems: elements belonging to right hand (based on names)

    Heuristic rules:
      - Right hand: link/joint names starting with "r_" OR the known mount joint "rgripper_mount"
      - Left hand: everything else belonging to the hand snippet
    """
    left = []
    right = []

    for el in list(hands_root):
        if el.tag not in ("link", "joint"):
            continue
        nm = el.attrib.get("name", "")

        is_right = nm.startswith("r_") or nm == "rgripper_mount"
        # many left items have no l_ prefix originally
        if is_right:
            right.append(el)
        else:
            left.append(el)

    if not left:
        raise RuntimeError("Did not detect any LEFT hand elements. Check that your hands URDF contains the left hand block.")
    if not right:
        raise RuntimeError("Did not detect any RIGHT hand elements. Check that your hands URDF contains the right hand block.")

    return left, right


def make_robot_wrapper(name="tmp"):
    return ET.Element("robot", {"name": name})


def main():
    # Load G1
    g1_tree = ET.parse(G1_URDF_IN)
    g1_root = g1_tree.getroot()

    # Remove stock hand stuff from G1
    remove_by_name(g1_root, "link", G1_REMOVE_LINKS)
    remove_by_name(g1_root, "joint", G1_REMOVE_JOINTS)

    # Load hands raw file
    hands_tree = ET.parse(HANDS_URDF_IN)
    hands_root = hands_tree.getroot()

    # Rewrite mesh paths inside the hands file (before splitting/prefixing)
    rewrite_mesh_paths(hands_root, HAND_ASSET_ROOT, USE_GLB_FOR_VISUAL)

    # Split elements
    left_elems, right_elems = extract_left_right_subtrees(hands_root)

    # Wrap each side into its own <robot> root so we can safely run transforms
    left_root = make_robot_wrapper("left_hand_tmp")
    right_root = make_robot_wrapper("right_hand_tmp")
    for el in left_elems:
        left_root.append(copy.deepcopy(el))
    for el in right_elems:
        right_root.append(copy.deepcopy(el))

    # -----------------------
    # LEFT: prefix names + retarget mount parent
    # -----------------------
    # In the raw snippet, left mount joint is "lgripper_mount"
    # but its parent link is "left_tool0". We want it to be LEFT_PARENT_LINK.
    #
    # Important: when we prefix, the mount joint name will become l_lgripper_mount
    # unless we exclude it. We'll prefix everything (including the mount), then retarget.
    prefix_names(left_root, LEFT_PREFIX)

    # Retarget the *prefixed* mount joint name
    retarget_mount_joint_parent(left_root, LEFT_PREFIX + "lgripper_mount", LEFT_PARENT_LINK)

    # Also: the mount joint currently has child link "hand_base_link" which becomes "l_hand_base_link" after prefix.
    # That’s fine.

    # -----------------------
    # RIGHT: retarget mount parent only
    # -----------------------
    # Right mount joint name in snippet: "rgripper_mount"
    # Parent in snippet: "right_tool0" -> RIGHT_PARENT_LINK
    retarget_mount_joint_parent(right_root, "rgripper_mount", RIGHT_PARENT_LINK)

    # -----------------------
    # Merge into G1
    # -----------------------
    for el in list(left_root):
        if el.tag in ("link", "joint"):
            g1_root.append(el)

    for el in list(right_root):
        if el.tag in ("link", "joint"):
            g1_root.append(el)

    # Write out
    g1_tree.write(G1_URDF_OUT, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote merged URDF: {G1_URDF_OUT}")
    print()
    print("Notes:")
    print(f" - Left hand got prefix '{LEFT_PREFIX}' for all its links/joints.")
    print(f" - Left mount joint '{LEFT_PREFIX}lgripper_mount' parent -> '{LEFT_PARENT_LINK}'")
    print(f" - Right mount joint 'rgripper_mount' parent -> '{RIGHT_PARENT_LINK}'")
    print(f" - Mesh filenames rewritten under: {HAND_ASSET_ROOT}/meshes/...")

if __name__ == "__main__":
    main()
