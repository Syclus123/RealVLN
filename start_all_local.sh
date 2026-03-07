#!/bin/bash

# config
ROBOT_IP="192.168.43.20"
ROBOT_USER="unitree"
ROBOT_PASS="123"
SESSION_NAME="vln_project"

SSH_CMD="sshpass -p $ROBOT_PASS ssh -o StrictHostKeyChecking=no $ROBOT_USER@$ROBOT_IP"

# start tmux session
tmux new-session -d -s $SESSION_NAME -n "nomachine"

# nomachine.sh
tmux send-keys -t $SESSION_NAME:0 "$SSH_CMD './nomachine.sh'" C-m

# network tunnel
tmux new-window -t $SESSION_NAME -n "ssh_tunnel"
TUNNEL_CMD="sshpass -p t0pxxqky ssh -N -L 5802:127.0.0.1:5802 -p 30438 root@183.147.142.40"
tmux send-keys -t $SESSION_NAME:1 "$SSH_CMD \"$TUNNEL_CMD\"" C-m

# robot start
tmux new-window -t $SESSION_NAME -n "start_robot"
tmux send-keys -t $SESSION_NAME:2 "$SSH_CMD 'cd InternNav-deploy/onboard/ && ./start_robot.sh'" C-m

# ROS2 Launch
tmux new-window -t $SESSION_NAME -n "ros2_launch"
tmux send-keys -t $SESSION_NAME:3 "$SSH_CMD 'ros2 launch go2_core go2_startup_XT16.launch.py'" C-m

# YOLO Client
tmux new-window -t $SESSION_NAME -n "yolo_client"
YOLO_CMD="python3 -u yolo_detect_client.py \
    --server-url http://127.0.0.1:5802/detect \
    --cam-json cam_params.json \
    --rgb-topic /camera/camera/color/image_raw \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    --odom-topic /odom_bridge \
    --frame-stride 10 \
    --goal-standoff 0.5 \
    --map-frame map \
    --base-link-frame base_link"
tmux send-keys -t $SESSION_NAME:4 "$SSH_CMD \"$YOLO_CMD\"" C-m

# enter session
tmux attach-session -t $SESSION_NAME