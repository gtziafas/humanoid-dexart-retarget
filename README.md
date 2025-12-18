Retargeting of **whole-body** human motion to humanoid robots for **dexterous** manipulation of **articulated** objects.

<p align="center">
  <img src="media/arctic_ketchup_snippet.gif" height="300" />
  <img src="media/recording.gif" height="300" />
</p>

Currently support [unitree G1](https://github.com/unitreerobotics/unitree_ros) with [Brainco hand](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_with_brainco_hand), using MoCap-based human-object interaction data. 

We implement the interaction-preserving Laplacian deformation objective from [OmniRetarget](https://arxiv.org/pdf/2509.26633), using the JAX-based Levenberg–Marquardt solver from [VideoMimic](https://arxiv.org/pdf/2505.03729), implemented within the [PyRoki](https://chungmin99.github.io/pyroki/) framework.

## Installation
Create a conda / mamba environment. We have tested the code with Python 3.10. Dependencies are self-contained within the `pyroki` framework:
```
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```
If you want to install JAX with CUDA support, you can use
```
pip install -U "jax[cuda12]"
```
for the the appropriate CUDA driver version in your machine.

## Usage
We support two versions, one with full-body supervision (SMPLX) from [ARCTIC](https://arctic.is.tue.mpg.de/), which also contains articulated object URDF models, and one with hands-only supervision (MANO) from the [H2O](https://github.com/taeinkwon/h2odataset) dataset.

### ARCTIC
Run:
```
python retarget_arctic.py --task_id=phone_use_01
```
Check `example_data/arctic` for all tasks. The data were generated from the "Sanity check" download section of ARCTIC from [here](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md). We further solve for a scale factor and global pose to align the SMPLX annotations for the hand joints with the corresponding MANO annotations, which are the ones calibrated with the object annotations in this dataset.

### H2O
```
python retarget_h2o.py --task_id=milk
```
Check `example_data/hoi` for all tasks. The data were downloaded from the "pose" version of the dataset. We further identify a reference frame aligned with the table (besides a z-axis offset) and express all annotations wrt. this frame. 

Since we only have MANO keypoints, and not for the entire body, we need some priors to ensure the robot maintains a human-like posture. We add adittional costs to implement this, but for now tend to be finicky, and their weights might require fine-tuning for each scene.

### Object Augmentations
You can activate the flag `Object Augmentation` in Viser to augment the initial pose of the object. The interaction mesh is expressed wrt. the local object frame, so retargeting works out-of-the-box for different initial object poses.

## TODOs

- [x] ARCTIC demo 
- [ ] Integrate more humanoid / hand models.
- [x] Make `viser` visualization more modular, add checkboxes / param boxes for object augmentations.
- [x] Hand-only retargeting demo, e.g. from DexYCB or HOI datasets.
