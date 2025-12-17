Retargeting of **whole-body** human motion to humanoid robots for **dexterous** manipulation of **articulated** objects.

<p align="center">
  <img src="media/arctic_ketchup_snippet.gif" height="300" />
  <img src="media/recording.gif" height="300" />
</p>

Currently support [unitree G1](https://github.com/unitreerobotics/unitree_ros) with [Brainco hand](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_with_brainco_hand), using example human-object interaction data from [ARCTIC](https://arctic.is.tue.mpg.de/). 

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
Run:
```
python retarget_arctic.py --task_id=phone_use_01
```
Check `example_data/arctic` for all tasks. The data were generated from the "Sanity check" download section of ARCTIC from [here](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md). We further solve for a scale factor and global pose to align the SMPLX annotations for the hand joints with the corresponding MANO annotations, which are the ones calibrated with the object annotations in this dataset.

You can activate the flag `--obj_augm` to augment the initial pose of the object (check script for augmentation params). The interaction mesh is expressed wrt. the local object frame, so retargeting works out-of-the-box for different initial object poses.
