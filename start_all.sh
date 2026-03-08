#!/bin/bash

# session name
SESSION="vln_run"

# check
tmux kill-session -t $SESSION 2>/dev/null

# 每个新终端先选择 ROS 版本（输入 1）
select_ros_version() {
    local target="$1"
    tmux send-keys -t "$target" "1" C-m
    sleep 0.3
}

run_in_target() {
    local target="$1"
    local cmd="$2"
    select_ros_version "$target"
    tmux send-keys -t "$target" "$cmd" C-m
}

# 4 宫格主窗口：tunnel / robot / ros2 / yolo 同屏显示（首个窗口同时创建 session）
tmux new-session -d -s "$SESSION" -n 'main'
tmux split-window -h -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION:0.1"
tmux select-layout -t "$SESSION:0" tiled

# pane 对应关系：
# 0.0 tunnel, 0.1 robot, 0.2 ros2, 0.3 yolo
run_in_target "$SESSION:0.0" "sshpass -p t0pxxqky ssh -N -L 5802:127.0.0.1:5802 -p 30438 root@183.147.142.40"
run_in_target "$SESSION:0.1" "cd ~/InternNav-deploy/onboard/ && ./start_robot.sh"
run_in_target "$SESSION:0.2" "ros2 launch go2_core go2_startup_XT16.launch.py"

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
run_in_target "$SESSION:0.3" "cd ~/yolo_deploy/ && $YOLO_CMD"

# enter session
tmux select-window -t "$SESSION:0"
tmux attach-session -t "$SESSION"