#!/bin/bash
# =============================================================================
# YOLO 目标检测系统启动示例脚本
# 包含 yolo_detect_server.py 和 yolo_detect_client.py 的完整参数说明
# =============================================================================

# -----------------------------------------------------------------------------
# 公共配置（根据实际环境修改）
# -----------------------------------------------------------------------------
CONDA_ENV="yolo"                        # conda 环境名
SERVER_HOST="127.0.0.1"
SERVER_PORT=5802
CAM_JSON="./cam_params.json"            # 相机参数文件（内含 fx/fy/cx/cy/scale/cam-offset-x/y/z）


# =============================================================================
# 一、启动 SERVER（yolo_detect_server.py）
# =============================================================================
# 功能：接收 HTTP /detect 请求，执行 YOLO 检测 + 3D 跟踪融合，返回 JSON 结果。
# 必须先于 client 启动。
#
# 最简启动（仅 RGBD，无保存）：
#   conda run -n $CONDA_ENV python -u yolo_detect_server.py \
#       --model yolov8s-world.pt \
#       --cam-json $CAM_JSON \
#       --vocab "box,chair,person" \
#       --port $SERVER_PORT
#
# 完整参数说明：
conda run -n "$CONDA_ENV" python -u yolo_detect_server.py \
    \
    # ── 模型 ──────────────────────────────────────────────────────────────
    --model yolov8s-world.pt        \
    # 权重文件路径。支持：
    #   yolov8n/s/m/l/x.pt          标准 COCO 检测（固定类别）
    #   yolov8s-world.pt            YOLO-World 开放词表（推荐，配合 --vocab 使用）
    #   yolo11l.pt                  YOLOv11
    \
    --vocab "box,chair,person"      \
    # 仅对 YOLO-World 有效，逗号分隔的目标类别词表。
    # 也可指向一个 txt 文件（每行一个类别）。
    # 不填则使用模型默认词表。
    \
    --conf 0.35                     \
    # 检测置信度阈值（0~1），低于此值的检测框丢弃。
    # 词表类别多时可适当调低（0.25~0.35）；精准场景可调高（0.5+）。
    \
    --classes "box,chair"           \
    # （可选）在 --vocab 基础上进一步过滤，只保留这几个类别。
    # 若不填则保留 --vocab 中全部类别。
    \
    # ── 相机内参 ──────────────────────────────────────────────────────────
    --cam-json "$CAM_JSON"          \
    # 相机参数 JSON 文件（推荐）。会自动读取：
    #   fx / fy / cx / cy           相机内参
    #   camera.scale                深度单位 scale（depth_m = raw / scale）
    #   cam-offset-x/y/z            相机相对 base_link 的平移偏移
    # 若不提供 cam-json，则使用下方独立参数：
    \
    # --fx 386.5 --fy 386.5         # 焦距（像素）
    # --cx 328.9 --cy 244.0         # 主点（像素）
    # --depth-scale 0.0001          # 深度单位到米的比例
    #                               # D435i 默认单位 0.1mm → scale=0.0001
    #                               # 标准 mm 单图 → scale=0.001
    \
    # ── 3D 跟踪与融合 ────────────────────────────────────────────────────
    --fusion moving_average         \
    # 世界坐标融合方式：
    #   none            每帧直接用原始坐标，无平滑
    #   moving_average  滑动平均（推荐，稳定）
    #   kalman          卡尔曼滤波（更平滑但有延迟）
    \
    --moving-avg-window 5           \
    # 滑动平均窗口大小（帧数）。越大越平滑但响应越慢。
    \
    --track-iou-thres 0.3           \
    # 帧间跟踪匹配 IoU 阈值（0~1）。
    # 低→宽松匹配（适合快速移动）；高→严格匹配（适合密集目标）。
    \
    --track-max-miss 8              \
    # track 最多连续丢失多少帧后删除。增大可保留短暂遮挡的目标。
    \
    --merge-distance-thres 0.35     \
    # 同类别 track 在世界坐标系距离（米）小于此值时合并为同一目标。
    # 设为 0 或负数则关闭合并。
    \
    --center-patch-radius 2         \
    # 目标中心深度取中位数的邻域半径（像素）。
    # 增大可减少噪声，但对小目标可能取到背景深度。
    \
    # ── 深度来源 ─────────────────────────────────────────────────────────
    --depth-source lidar            \
    # 深度测距来源：
    #   rgbd    使用 D435i 等 RGBD 相机深度图（默认）
    #   lidar   优先使用激光雷达点云：
    #             1. 先从逐像素投影深度图取中心 patch 中位数
    #             2. 失败 → 框内所有投影点取中位数
    #             3. 再失败 → 回退到 RGBD 深度图
    \
    --lidar-min-pts 1               \
    # lidar 模式下检测框内有效点数下限，低于此值回退到 RGBD。
    \
    # ── 保存模式 ─────────────────────────────────────────────────────────
    --output-dir ./output_realtime  \
    # 开启保存模式，指定输出根目录（自动用时间戳创建子目录）。
    # 不填则不保存任何文件。
    \
    --save-vis                      \
    # （需 --output-dir）逐帧保存带标注的 RGB 和深度图：
    #   output_dir/vis/rgb/frameXXXXXX.jpg
    #   output_dir/vis/depth/frameXXXXXX.jpg
    \
    --save-world-plot               \
    # （需 --output-dir）Ctrl+C 退出时保存世界坐标俯视轨迹图：
    #   output_dir/world_tracks.png
    \
    --world-plot-min-points 3       \
    # 轨迹图中 track 至少需要多少个点才绘制（过滤噪声 track）。
    \
    # ── 服务地址 ─────────────────────────────────────────────────────────
    --host 0.0.0.0                  \
    # 监听地址：0.0.0.0 表示接受任意来源；127.0.0.1 表示仅本机。
    \
    --port $SERVER_PORT
    # 监听端口（默认 5802）。


# =============================================================================
# 二、启动 CLIENT（yolo_detect_client.py）
# =============================================================================
# 功能：ROS2 节点，订阅 RGB+Depth+Odom（+可选激光雷达），
#       逐帧发送给 server，并提供交互式语义导航目标发布。
# 必须在 ROS2 环境中运行（source /opt/ros/<distro>/setup.bash）。
#
# 最简启动（仅 RGBD，无激光雷达）：
#   conda run -n $CONDA_ENV python -u yolo_detect_client.py \
#       --server-url http://$SERVER_HOST:$SERVER_PORT/detect \
#       --cam-json $CAM_JSON
#
# 完整参数说明：
conda run -n "$CONDA_ENV" python -u yolo_detect_client.py \
    \
    # ── Server 地址 ──────────────────────────────────────────────────────
    --server-url "http://${SERVER_HOST}:${SERVER_PORT}/detect" \
    # Server 的 /detect 接口完整 URL。
    \
    # ── ROS2 话题 ────────────────────────────────────────────────────────
    --rgb-topic   /camera/camera/color/image_raw                \
    # RGB 图像话题（sensor_msgs/Image）。
    \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    # 深度图话题（sensor_msgs/Image，16UC1，与 RGB 对齐）。
    # D435i 使用已对齐深度：aligned_depth_to_color
    \
    --odom-topic  /odom_bridge                                  \
    # 里程计话题（nav_msgs/Odometry）。
    # 用于将每帧相机位姿（T_w_cam）发给 server。
    # 典型值：/odom / /odom_bridge / /Odometry
    \
    # ── 帧率控制 ─────────────────────────────────────────────────────────
    --rate 0.3                      \
    # 两次 HTTP 请求之间的最小间隔（秒）。
    # 0.3 → 最高约 3 fps 发送（实际受 server 推理速度限制）。
    \
    --frame-stride 10               \
    # 帧采样间隔：每隔 N 帧处理一帧。
    # 10 → 摄像头 30fps 时约 3fps 发送（与 --rate 共同控制实际发送频率）。
    # 1  → 每帧都发送（低延迟场景）。
    \
    # ── 相机偏移（base_link → 相机） ─────────────────────────────────────
    --cam-json "$CAM_JSON"          \
    # 从 JSON 读取 cam-offset-x/y/z（相机在 base_link 坐标系下的平移偏移）。
    # 用于将里程计(base_link)位姿转换为相机位姿后发给 server。
    \
    # 也可直接指定偏移（优先级高于 --cam-json）：
    # --cam-offset-x 0.15           # 相机在 base_link 前方 0.15m
    # --cam-offset-y 0.0
    # --cam-offset-z 0.30           # 相机在 base_link 上方 0.30m
    \
    # ── 语义导航参数 ─────────────────────────────────────────────────────
    --goal-topic    /goal_pose      \
    # 发布导航目标的话题（geometry_msgs/PoseStamped）。
    # /goal_pose 为 Nav2 默认接口，直接被 navigate_to_pose action 消费。
    \
    --map-frame     map             \
    # 地图坐标系 frame id，用于 TF 查询机器人当前位置（map → base_link）。
    \
    --base-link-frame base_link     \
    # 机器人底盘 frame id（TF 查询用）。
    \
    --goal-standoff 0.6             \
    # 导航目标距目标物体的停留距离（米）。
    # 例：目标在 2m 处，standoff=0.6，则机器人导航到距目标 0.6m 的位置。
    \
    --z-offset 0.0                  \
    # 导航目标 z 方向额外偏移（米），通常保持 0。
    \
    # ── 激光雷达（可选，配合 server --depth-source lidar 使用） ───────────
    --lidar-topic /livox/lidar      \
    # 激光雷达话题。不填则不订阅激光雷达。
    # 2D 激光雷达：/scan（LaserScan）
    # 3D 激光雷达：/livox/lidar 或 /velodyne_points（PointCloud2）
    \
    --lidar-type pointcloud2        \
    # 激光雷达消息类型：
    #   laserscan   2D 激光雷达（sensor_msgs/LaserScan）
    #   pointcloud2 3D 激光雷达（sensor_msgs/PointCloud2）
    \
    --camera-frame camera_color_optical_frame \
    # 相机光学坐标系的 TF frame id，用于查询 T_cam_lidar 静态变换。
    # D435i 默认：camera_color_optical_frame
    # 该变换只查询一次后缓存，无额外开销。
    \
    --lidar-max-points 2000
    # 每帧发送给 server 的最大激光雷达点数（超出则随机下采样）。
    # 增大可提高深度精度，但增加 HTTP 传输开销。


# =============================================================================
# 三、实用启动命令（可直接复制使用）
# =============================================================================

# ── 场景1：仅 RGBD，快速测试 ──────────────────────────────────────────────
# 终端1：
# conda run -n yolo python -u yolo_detect_server.py \
#     --model yolov8s-world.pt --cam-json cam_params.json \
#     --vocab "box,chair" --conf 0.35 --depth-source rgbd --port 5802
#
# 终端2：
# conda run -n yolo python -u yolo_detect_client.py \
#     --server-url http://127.0.0.1:5802/detect \
#     --cam-json cam_params.json --frame-stride 10


# ── 场景2：使用 3D 激光雷达（Livox）+ 保存结果 ────────────────────────────
# 终端1：
conda run -n yolo python -u yolo_detect_server.py \
    --model yolov8s-world.pt --cam-json cam_params.json \
    --vocab "box,chair,person" --conf 0.35 \
    --depth-source lidar --lidar-min-pts 2 \
    --output-dir ./output --save-vis --save-world-plot --port 5802

# 终端2：
conda run -n yolo python -u yolo_detect_client.py \
    --server-url http://127.0.0.1:5802/detect \
    --cam-json cam_params.json \
    --lidar-topic /xt16_cloud --lidar-type pointcloud2 \
    --camera-frame camera_color_optical_frame \
    --frame-stride 10 --goal-standoff 0.4


# ── 场景3：使用 2D 激光雷达（/scan）────────────────────────────────────────
# 终端2（server 同场景2，depth-source lidar）：
# conda run -n yolo python -u yolo_detect_client.py \
#     --server-url http://127.0.0.1:5802/detect \
#     --cam-json cam_params.json \
#     --lidar-topic /scan --lidar-type laserscan \
#     --camera-frame camera_color_optical_frame \
#     --frame-stride 10


# ── 交互式导航命令（client 启动后在终端输入）─────────────────────────────
# 目标类别 > box        → 查询 "box"，自动发布导航目标到 /goal_pose
# 目标类别 > list       → 列出 server 当前所有已追踪类别
# 目标类别 > q          → 退出交互（检测继续运行）
