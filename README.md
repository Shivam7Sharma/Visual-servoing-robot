This folder contains the encoderless robots visual servoing implementation.
Skeleton Code for VB Manip class

vbmbot.xacro - Robot description file for 2 link planar manipulator

vbmbot.gazebo - References for xacro

materials.xacro - common gazebo material descriptions

Nodes - vbmbot_joint_controller.py - launches with vb_joint_vel_and_pos_pub.launch

Run this project by launching the following - roslaunch vision_based_manipulation vb_publish_joint_pose_and_vel.launch -- When node is initialized, the robot is in home position as send by the launch file. the position controller publishes a target joint position that moves the arm from home position, then ros switches controller to publish random joint velocity to move the arm.