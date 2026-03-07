#!/bin/bash

# session name
SESSION="vln_run"

# check
tmux kill-session -t $SESSION 2>/dev/null

# nomachine
tmux new-session -d -s $SESSION -n 'nomachine'
tmux send-keys -t $SESSION:0 "./nomachine.sh" C-m

# network tunnel
tmux new-window -t $SESSION -n 'tunnel'
tmux send-keys -t $SESSION:1 "sshpass -p t0pxxqky ssh -N -L 5802:127.0.0.1:5802 -p 30438 root@183.147.142.40" C-m

# robot start
tmux new-window -t $SESSION -n 'robot'
tmux send-keys -t $SESSION:2 "cd ~/InternNav-deploy/onboard/ && ./start_robot.sh" C-m

# ROS2 Launch
tmux new-window -t $SESSION -n 'ros2'
tmux send-keys -t $SESSION:3 "ros2 launch go2_core go2_startup_XT16.launch.py" C-m

# YOLO Client
tmux new-window -t $SESSION -n 'yolo'
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
tmux send-keys -t $SESSION:4 "cd ~/InternNav-deploy/onboard/ && $YOLO_CMD" C-m

# enter session
tmux select-window -t $SESSION:0
tmux attach-session -t $SESSION