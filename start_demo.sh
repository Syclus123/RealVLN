#!/bin/bash
# ============================================================
#  一键启动脚本：语音通话 + VLN 全栈
#
#  两个独立 tmux session，互不干扰：
#
#  voice session (voice_run):
#    单 pane: 完全按照 run_go2_fast.sh 方式启动语音功能
#             (在 tmux 内拥有独立终端，ALSA 音频设备稳定)
#
#  vln session (vln_run):
#    pane 0: SSH 隧道 (端口转发 5802)
#    pane 1: 机器人底层驱动 (start_robot.sh)
#    pane 2: ROS2 导航栈 (go2_startup)
#    pane 3: YOLO 检测客户端
# ============================================================

set -e

SESSION_VLN="vln_run"
SESSION_VOICE="voice_run"
GO2_FAST_DIR="$HOME/RealVLN/Go2RealtimeAPI/master/RealtimeAPI-sdk-python"

# ── 0. 清理上一次运行 ──────────────────────────
tmux kill-session -t "$SESSION_VLN"   2>/dev/null || true
tmux kill-session -t "$SESSION_VOICE" 2>/dev/null || true
pkill -f "src\.audio_server\.server" 2>/dev/null || true
pkill -f "src\.main_client_fast"     2>/dev/null || true
sleep 1

rm -f "$GO2_FAST_DIR/audio_server.log" \
      "$GO2_FAST_DIR/audio_client.log" \
      "$GO2_FAST_DIR/main_client.log" \
      "$GO2_FAST_DIR/nohup.out.local_server" \
      "$GO2_FAST_DIR/nohup.out.fast_client"
mkdir -p "$GO2_FAST_DIR/nav_logs"
: > "$GO2_FAST_DIR/nav_logs/navigation.jsonl"

echo "========================================"
echo "  VLN + Voice 一键启动"
echo "========================================"

# ── helper ────────────────────────────────────
select_ros_version() {
    tmux send-keys -t "$1" "1" C-m
    sleep 0.3
}
run_in_target() {
    select_ros_version "$1"
    tmux send-keys -t "$1" "$2" C-m
}

# ── 1. 语音服务 (独立 tmux session) ──────────
#    直接在 tmux pane 内执行 run_go2_fast.sh，与手动运行完全一致
#    语音进程拥有独立终端上下文，避免 ALSA fd 失效
echo "[Voice] 创建 tmux session: $SESSION_VOICE ..."
tmux new-session -d -s "$SESSION_VOICE" -n 'voice'
select_ros_version "$SESSION_VOICE:0.0"
tmux send-keys -t "$SESSION_VOICE:0.0" \
    "cd $GO2_FAST_DIR && bash run_go2_fast.sh" C-m

echo "[Voice] ✓ 语音启动命令已发送至 tmux session: $SESSION_VOICE"
echo ""

# ── 2. VLN 服务 (独立 tmux session) ──────────
echo "[VLN] 创建 tmux session: $SESSION_VLN ..."
tmux new-session -d -s "$SESSION_VLN" -n 'vln'
tmux split-window -h -t "$SESSION_VLN:0.0"
tmux split-window -v -t "$SESSION_VLN:0.0"
tmux split-window -v -t "$SESSION_VLN:0.1"
tmux select-layout -t "$SESSION_VLN:0" tiled

# pane 0: SSH 隧道
run_in_target "$SESSION_VLN:0.0" \
    "sshpass -p t0pxxqky ssh -N -L 5802:127.0.0.1:5802 -p 30438 root@183.147.142.40"

# pane 1: 机器人底层驱动
run_in_target "$SESSION_VLN:0.1" \
    "cd ~/InternNav-deploy/onboard/ && ./start_robot.sh"

# pane 2: ROS2 导航栈
run_in_target "$SESSION_VLN:0.2" \
    "source ~/ros2_ws/install/setup.bash; ros2 launch go2_core go2_startup_MID360.launch.py"

# pane 3: YOLO 检测客户端
YOLO_CMD="python3 -u yolo_detect_client.py \
    --server-url http://127.0.0.1:5802/detect \
    --cam-json cam_params.json \
    --rgb-topic /camera/camera/color/image_raw \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    --odom-topic /odom_bridge \
    --frame-stride 10 \
    --goal-standoff 0.5 \
    --map-frame map \
    --base-link-frame base_link \
    --webui-port 8083 \
    --goal-send-mode goal_pose \
    --arrival-threshold 0.5 \
    --nav-log $GO2_FAST_DIR/nav_logs/navigation.jsonl \
    --go2-interface eth0"
run_in_target "$SESSION_VLN:0.3" "cd ~/RealVLN/ && $YOLO_CMD"

# ── 3. 提示信息 ──────────────────────────────
echo ""
echo "========================================"
echo "  两个 tmux session 已创建："
echo ""
echo "  [语音] tmux attach -t $SESSION_VOICE"
echo "  [导航] tmux attach -t $SESSION_VLN"
echo ""
echo "  Ctrl+B d 脱离当前 session"
echo ""
echo "  [停止所有服务]"
echo "    tmux kill-session -t $SESSION_VLN"
echo "    tmux kill-session -t $SESSION_VOICE"
echo "========================================"

# 默认 attach 到导航 session
tmux attach-session -t "$SESSION_VLN"
