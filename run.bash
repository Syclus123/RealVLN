client:
--------------------------------------------
ssh unitree@192.168.43.20
./nomachine.sh
--------------------------------------------
rviz2
--------------------------------------------
Camera odometer:
cd InternNav-deploy/onboard/
./start_robot.sh
--------------------------------------------
net:
ssh -L 5802:127.0.0.1:5802 -p 1021 root@8.130.45.234 -N

ssh -N \
  -L 5802:127.0.0.1:5802 \
  -p 30438 root@183.147.142.40

nmcli dev wifi connect "A" password "12345678"

test wifi:
nmcli device wifi list
--------------------------------------------
nav2:
source ~/ros2_ws/install/setup.bash
ros2 launch go2_core go2_startup_XT16.launch.py
--------------------------------------------
free:
t0pxxqky
--------------------------------------------
client query:
cd yolo_deploy
python3 -u yolo_detect_client.py \
    --server-url http://127.0.0.1:5802/detect \
    --cam-json cam_params.json \
    --rgb-topic /camera/camera/color/image_raw \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    --odom-topic /odom_bridge \
    --frame-stride 10 \
    --goal-standoff 0.5 \
    --map-frame map \
    --base-link-frame base_link \
    --go2-interface eth0

go2:
--go2-interface eth0

--------------------------------------------
server:
python -u yolo_detect_server.py \
    --model yolov8s-world.pt \
    --cam-json cam_params.json \
    --vocab chair,bag,box --conf 0.35 \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --cam-json cam_params.json \
    --vocab "chair,bag,box,trash can,bottle" --conf 0.35 \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

1/
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --cam-json cam_params.json \
    --vocab "chair,bag,box" --conf 0.35 \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

2/
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --cam-json cam_params.json \
    --vocab "chair,bag,box,sofa,plant,desk,bottle,potted plant,trash can,bin,bed,fridge,elevator,elevator door,umbrella,toilet,table,refrigerator" --conf 0.35 \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --live-vis --live-vis-depth --live-vis-scale 1.0 \
    --port 5802

python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --cam-json cam_params.json \
    --vocab "chair,bag,box,sofa,plant,desk,bottle,potted plant,trash can,bin,bed,fridge,elevator,elevator door,door,umbrella,table,refrigerator," --conf 0.35 \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

python -u yolo_detect_server.py \
    --model yolo26x.pt \
    --cam-json cam_params.json \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

open:
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --vocab yolo_vocab/obj365v1_class_texts.json \
    --conf 0.35 \
    --cam-json cam_params.json \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

deepseek intent recognition:
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --vocab yolo_vocab/obj365v1_class_texts.json \
    --conf 0.35 \
    --cam-json cam_params.json \
    --fusion moving_average --moving-avg-window 5 \
    --deepseek-api-key sk-xxxxxxxxxxxxxxxx \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802

caption:
export ARK_API_KEY=f622fb17-c76d-48d4-a184-17f52f7e41aa
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --vocab yolo_vocab/obj365v1_class_texts.json \
    --conf 0.35 \
    --cam-json cam_params.json \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802 \
    --enable-caption

export ARK_API_KEY=f622fb17-c76d-48d4-a184-17f52f7e41aa
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --vocab basketball,soccer,tissue,toy,helmet,can \
    --conf 0.35 \
    --cam-json cam_params.json \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802 \
    --enable-caption



---------------------------------------------------------
Lidar

server
python -u yolo_detect_server.py \
    --model yolov8x-worldv2.pt \
    --cam-json cam_params.json \
    --vocab chair,bag,box --conf 0.35 \
    --fusion moving_average --moving-avg-window 5 \
    --merge-distance-thres 0.2 \
    --output-dir ./output_realtime \
    --save-vis \
    --save-world-plot \
    --port 5802 \
    --depth-source lidar --lidar-min-pts 2


client:
python3 -u yolo_detect_client.py \
    --server-url http://127.0.0.1:5802/detect \
    --cam-json cam_params.json \
    --rgb-topic /camera/camera/color/image_raw \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    --odom-topic /odom_bridge \
    --frame-stride 10 \
    --goal-standoff 0.5 \
    --map-frame map \
    --base-link-frame base_link \
    --lidar-topic /xt16_cloud --lidar-type pointcloud2
