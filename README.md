# Consensus-Driven-Uncertainty
**Consensus-Driven Uncertainty for Robotic Grasping based on RGB Perception**

Eric C. Joyce, Qianwen Zhao, Nathaniel Burgdorfer, Long Wang, Philippos Mordohai

Accepted to [**IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025**](https://www.iros25.org/)

[![Video poster](https://www.ericjoycefilm.com/reel/_2024/consensus-driven-uncertainty/obj/img/poster.jpg)](https://www.ericjoycefilm.com/reel/_2024/consensus-driven-uncertainty/)

## Predict 6-DoF Object Poses from an RGB Image
The directory `Pose-Estimators` contains instructions and resources for running the three pose-estimators used in our study.

Alternatively, you can use pre-computed pose estimates found in `Pose-Estimates`.

`BOP-Metrics` contains pose-estimate measurements. Many of these are defined by the BOP Benchmark. This folder also contains metrics for pose-estimators that do not appear in the paper.

## Conduct Grasping Trials in a Physics Simulator
The directories `Parallel-Gripper` and `Underactuated-Gripper` contain the results of grasping trials conducted in simulation using the parallel and underactuated effectors, respectively.

## Train a Deep Network to Predict Grasp Success
`Train-Grasp-Success-Prediction` contains the resources necessary to train grasp success-prediction networks according to our method. The data split and baseline used in our study can also be reproduced.
