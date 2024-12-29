import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2 as cv
from cv2 import aruco
import numpy as np
import pymavlink.mavutil as utility
from pymavlink import mavutil
import pymavlink.dialects.v20.all as dialect
import math, sys, time

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(Image,'/camera/image',self.listener_callback,10)
        self.subscription  # 防止未使用變量警告
        self.br = CvBridge()  # 初始化 CvBridge 用於轉換 ROS 影像訊息

        # 連接到飛機
        self.vehicle = utility.mavlink_connection('udpin:127.0.0.1:14550')
        self.vehicle.wait_heartbeat()
        print("Connected to system:", self.vehicle.target_system, ", component:", self.vehicle.target_component)

        # 清除原有任務
        self.clear_mission()

        # 更改 Auto 模式
        self.mode_change_to_Auto()

        # 上傳任務點
        self.set_mission_points()

        # 確認任務總數
        self.get_waypoint_count()

        # 設置 WP_YAW_BEHAVIOR 為 1 （跟著航向）
        self.set_wp_yaw_behavior(1)
        print("WP_YAW_BEHAVIOR set to 1")

        # 其他參數設置
        self.id_to_find_16 = 16
        self.id_to_find_0 = 0
        self.marker_size_16 = 500  # 單位公分
        self.marker_size_0 = 100    # 單位公分

        #解鎖
        self.arm()
    
    def listener_callback(self, msg):
        # 检查当前飞行模式是否为 GUIDED

        self.get_waypoint_info()

        if self.wp_point == self.wp_point_count:

            # 检查当前飞行模式是否为 GUIDED
            if self.flight_mode_name ==  "GUIDED":

                #print("飛行模式正確，開始進行 ArUco 定位")

                #-- 設置字體
                font = cv2.FONT_HERSHEY_SIMPLEX

                # 日誌訊息，提示正在接收影像幀
                #self.get_logger().info('Receiving video frame')

                # 使用 CvBridge 將 ROS 影像訊息轉換為 OpenCV 格式
                current_frame = self.br.imgmsg_to_cv2(msg)

                # 設置內部參數
                self.camera_matrix = np.array([[530.8269276712998, 0.0, 320.5],
                                            [0.0, 530.8269276712998, 240.5],
                                            [0.0, 0.0, 1.0]], dtype=np.float32)
                self.dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                cam_mat = np.array(self.camera_matrix)
                dist_coef = np.array(self.dist_coeff)
                
                # 獲取預定義的 ArUco 字典
                aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

                # 設定 ArUco 檢測參數
                parameters = aruco.DetectorParameters()
                parameters.adaptiveThreshWinSizeMin = 3
                parameters.adaptiveThreshWinSizeMax = 23
                parameters.adaptiveThreshWinSizeStep = 10
                parameters.minMarkerPerimeterRate = 0.03
                parameters.maxMarkerPerimeterRate = 4.0

                # 將畫面轉成灰階
                gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                #找到圖像中的所有aruco標記
                corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)

                if ids is not None:
                    for i, self.id in enumerate(ids):
                        # 確認 ArUco 標記的類型和大小
                        if self.id == self.id_to_find_16:
                            marker_size = self.marker_size_16  # 設置大標記的大小
                        elif self.id == self.id_to_find_0:
                            marker_size = self.marker_size_0   # 設置小標記的大小
                        else:
                            continue  # 如果不是指定的標記，則跳過處理

                        # 估算 ArUco 標記的姿態和位置
                        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_size, self.camera_matrix, self.dist_coeff)

                        # 提取標記的角點位置
                        corner_points = corners[i].reshape(4, 2)
                        corner_points = corner_points.astype(int)

                        top_right = corner_points[0].ravel()
                        top_left = corner_points[1].ravel()
                        bottom_right = corner_points[2].ravel()
                        bottom_left = corner_points[3].ravel()

                        # 從飛機獲取當前高度數據
                        altitude_data = self.vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
                        self.altitude = altitude_data.relative_alt / 1000.0 if altitude_data else 0

                        # 計算飛機到標記的距離
                        distance = np.sqrt(tVec[0][0][2] ** 2 + tVec[0][0][0] ** 2 + tVec[0][0][1] ** 2)

                        # 繪製標記的座標軸
                        cv2.drawFrameAxes(current_frame, self.camera_matrix, self.dist_coeff, rVec[0], tVec[0], 4)

                        # 計算飛機相對於標記的位置並顯示
                        pos_drone = np.array([tVec[0][0][1], -tVec[0][0][0], tVec[0][0][2]])  # 轉換為飛機座標
                        #str_position_drone = f"Drone Position (relative to Marker {ids[0]}) x={pos_drone[0]:.2f} y={pos_drone[1]:.2f} z={pos_drone[2]:.2f}"
                        #cv2.putText(current_frame, str_position_drone, (0, 100 + i * 100), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1, cv2.LINE_AA)

                        # 在標記上方顯示標記的 ID 和距離
                        cv2.putText(current_frame, f"id: {ids[0]} Alt: {round(self.altitude, 2)}m", top_right, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)

                        # 在標記右下方顯示飛機相對於標記的坐標位置
                        cv2.putText(current_frame, f"x:{round(pos_drone[0], 1)} y: {round(pos_drone[1], 1)}", bottom_right, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)

                        # 將 x y 座標存到其他值
                        self.x = pos_drone[0]
                        self.y = pos_drone[1]

                        print("x:", {round(pos_drone[0], 1)}, "y:", {round(pos_drone[1], 1)}, "z:", self.altitude)

                        # 呼叫精準降落函數
                        self.precise_landing()

                elif ids is None:
                    altitude_data = self.vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
                    self.altitude = altitude_data.relative_alt / 1000.0 if altitude_data else 0
                    self.not_found_aruco()

                # 顯示處理過的影像
                #cv2.imshow("camera", current_frame)
                cv2.waitKey(1)

            else:
                # 切換guided模式
                self.mode_change_to_guided()
                time.sleep(2)

                # 設置 WP_YAW_BEHAVIOR 為 0 （固定航向）
                self.set_wp_yaw_behavior(0)
                print("WP_YAW_BEHAVIOR set to 0")

        else:
            print("還沒達到最後任務點")
            pass
        
    # 定義任務點函數
    def set_mission_points(self):

        # 得到 home point 位置
        home_position = self.vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=True)

        # 獲得 home point 經度、緯度、高度
        latitude_of_home = home_position.lat / 1e7
        longitude_of_home = home_position.lon / 1e7

        # 創建任務項目列表
        target_locations = ((-35.36325649, 149.16578817, 10.0),
                            (-35.36321446, 149.16529466, 10.0),
                            (-35.36321446, 149.16529466, 10.0))      #latitude_of_home, longitude_of_home, 10.0)

        # 創建任務計數消息
        message = dialect.MAVLink_mission_count_message(target_system=self.vehicle.target_system,
                                                        target_component=self.vehicle.target_component,
                                                        count=len(target_locations) + 2,
                                                        mission_type=dialect.MAV_MISSION_TYPE_MISSION)

        # 發送任務計數消息給飛機
        self.vehicle.mav.send(message)

        # 這個循環會持續到接收到有效的MISSION_ACK消息
        while True:
            # 接收一條消息
            message = self.vehicle.recv_match(blocking=True)

            # 將消息轉換為字典
            message = message.to_dict()

            # 檢查這條消息是MISSION_REQUEST
            if message["mavpackettype"] == dialect.MAVLink_mission_request_message.msgname:
                print(message)

                # 檢查這個請求是任務項目
                if message["mission_type"] == dialect.MAV_MISSION_TYPE_MISSION:

                    # 獲取請求的任務項目序列號
                    seq = message["seq"]

                    # 創建任務項目整數消息
                    if seq == 0:
                        # 創建包含家庭位置的任務項目整數消息（第0個任務項目）
                        message = dialect.MAVLink_mission_item_int_message(target_system=self.vehicle.target_system,
                                                                        target_component=self.vehicle.target_component,
                                                                        seq=seq,
                                                                        frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                                                        command=dialect.MAV_CMD_NAV_WAYPOINT,
                                                                        current=0,
                                                                        autocontinue=0,
                                                                        param1=0,
                                                                        param2=0,
                                                                        param3=0,
                                                                        param4=0,
                                                                        x=0,
                                                                        y=0,
                                                                        z=0,
                                                                        mission_type=dialect.MAV_MISSION_TYPE_MISSION)
                    elif seq == 1:
                        # 創建包含起飛命令的任務項目整數消息（第1個任務項目）
                        message = dialect.MAVLink_mission_item_int_message(target_system=self.vehicle.target_system,
                                                                        target_component=self.vehicle.target_component,
                                                                        seq=seq,
                                                                        frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                                                        command=dialect.MAV_CMD_NAV_TAKEOFF,
                                                                        current=0,
                                                                        autocontinue=0,
                                                                        param1=0,
                                                                        param2=0,
                                                                        param3=0,
                                                                        param4=0,
                                                                        x=0,
                                                                        y=0,
                                                                        z=target_locations[0][2],
                                                                        mission_type=dialect.MAV_MISSION_TYPE_MISSION)
                    else:
                        # 創建包含目標位置的任務項目整數消息（其他任務項目）
                        message = dialect.MAVLink_mission_item_int_message(target_system=self.vehicle.target_system,
                                                                        target_component=self.vehicle.target_component,
                                                                        seq=seq,
                                                                        frame=dialect.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                                                                        command=dialect.MAV_CMD_NAV_WAYPOINT,
                                                                        current=0,
                                                                        autocontinue=0,
                                                                        param1=0,
                                                                        param2=0,
                                                                        param3=0,
                                                                        param4=0,
                                                                        x=int(target_locations[seq-2][0] * 1e7),
                                                                        y=int(target_locations[seq-2][1] * 1e7),
                                                                        z=target_locations[seq-2][2],
                                                                        mission_type=dialect.MAV_MISSION_TYPE_MISSION)

                    # 發送任務項目整數消息給飛機
                    self.vehicle.mav.send(message)

            # 檢查這條消息是MISSION_ACK
            elif message["mavpackettype"] == dialect.MAVLink_mission_ack_message.msgname:
                
                # 檢查這個確認消息是任務且已接受
                if message["type"] == dialect.MAV_MISSION_ACCEPTED and message["mission_type"] == dialect.MAV_MISSION_TYPE_MISSION:

                    # 打印上傳任務成功
                    print("Mission upload is successful")
                    break

    # 定義降落函數
    def arm(self):

        VEHICLE_ARM = 1

        # vehicle arm message
        vehicle_arm_message = dialect.MAVLink_command_long_message(
            target_system = self.vehicle.target_system,
            target_component = self.vehicle.target_component,
            command=dialect.MAV_CMD_COMPONENT_ARM_DISARM,
            confirmation=0,
            param1=VEHICLE_ARM,
            param2=0,
            param3=0,
            param4=0,
            param5=0,
            param6=0,
            param7=0
        )

        # check the pre-arm 
        while True:

            # observe the SYS_STATUS message
            message = self.vehicle.recv_match(type = dialect.MAVLink_sys_status_message.msgname, blocking = True)

            # convert to dictionary
            message = message.to_dict()

            # get sensor health
            onboard_control_sensors_health = message['onboard_control_sensors_health']

            # get pre-arm healthy bit
            prearm_status = onboard_control_sensors_health & dialect.MAV_SYS_STATUS_PREARM_CHECK == dialect.MAV_SYS_STATUS_PREARM_CHECK

            if  prearm_status:

                # vehicle can be armable
                print("vehicle is armable")

                # break the prearm check loop
                break
        
        while True:
            # arm the vehicle
            print("vehicle is arming....")  

            # send arm message
            self.vehicle.mav.send(vehicle_arm_message)

            # wait for COMMAND_ACK MESSAGE
            message = self.vehicle.recv_match(type=dialect.MAVLink_command_ack_message.msgname, blocking=True)

            # convert the message to dictionary
            message = message.to_dict()

            # check if the vehicle is armed
            if message['result'] == dialect.MAV_RESULT_ACCEPTED:
                
                # print that vehicle is armed
                print("Vehicle is armed!")
                break  # 跳出循环
            else:
                # print that vehicle is not armed
                print("Vehicle is not armed!")
                time.sleep(2)  # 等待1秒後再次發送ARM消息
                continue

    # 定義自動模式函數
    def mode_change_to_Auto(self):

        # desired flight mode
        FLIGHT_MODE = 'AUTO'

       # get supported flight modes
        flight_modes = self.vehicle.mode_mapping()
    
        # check the desired flight mode is supported
        if FLIGHT_MODE not in flight_modes.keys():

            # inform user that desired mode is not supported by the vehicle
            print(FLIGHT_MODE, 'is not supported')

            # exit the code
            exit()
        
        # create change mode message
        set_mode_message = dialect.MAVLink_command_long_message(
            target_system=self.vehicle.target_system,
            target_component=self.vehicle.target_component,
            command=dialect.MAV_CMD_DO_SET_MODE,
            confirmation=0,
            param1=dialect.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            param2=flight_modes[FLIGHT_MODE],
            param3=0,
            param4=0,
            param5=0,
            param6=0,
            param7=0
        )
        # catch HEARTBEAT message
        message = self.vehicle.recv_match(type=dialect.MAVLink_heartbeat_message.msgname, blocking=True)

        # convert this message to dictionary
        message = message.to_dict()

        # get the mode id
        mode_id = message['custom_mode']

        # get mode name
        flight_mode_names = list(flight_modes.keys())
        flight_mode_ids = list(flight_modes.values())
        flight_mode_index = flight_mode_ids.index(mode_id)
        self.flight_mode_name = flight_mode_names[flight_mode_index ]

        #print heartbeat message
        print("Mode name before:", self.flight_mode_name)

        # change flight mode
        self.vehicle.mav.send(set_mode_message)

        # do below always
        while True:

            # catch COMMAND_ACK message
            message = self.vehicle.recv_match(type=dialect.MAVLink_command_ack_message.msgname, blocking=True)

            # convert this message to dictionary
            message = message.to_dict()

            # check is the COMMAND_ACK is for DO_SET_MODE
            if message['command'] == dialect.MAV_CMD_DO_SET_MODE:

                # check the command is accepted or not
                if message['result'] == dialect.MAV_RESULT_ACCEPTED:

                    # inform the user
                    print("Changeing mode to", FLIGHT_MODE, "accepted from the vehicle")

                # not accepted
                else:

                    # inform the user
                    print("Changeing mode to", FLIGHT_MODE, "failed")

                # break the loop
                break

        # catch HEARTBEAT message
        message = self.vehicle.recv_match(type=dialect.MAVLink_heartbeat_message.msgname, blocking=True)

        # convert this message to dictionary
        message = message.to_dict()

        # get the mode id
        mode_id = message['custom_mode']

        # get mode name
        flight_mode_names = list(flight_modes.keys())
        flight_mode_ids = list(flight_modes.values())
        flight_mode_index = flight_mode_ids.index(mode_id)
        self.flight_mode_name = flight_mode_names[flight_mode_index ]

        #print heartbeat message
        print("Mode name after:", self.flight_mode_name)

        # change flight mode
        self.vehicle.mav.send(set_mode_message)

    # 定義精確降落函數
    def precise_landing(self):

        # 確認 ArUco 標記的類型
        if self.id == self.id_to_find_16:
            print("Find aruco marker 16")

            # 確認 ArUco 距離飛機距離
            if self.x > 100 and self.y > 100:
                print("ArUco 在 x 正方向且 y 正方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,-1,-1,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            elif self.x < -100 and self.y > 100:
                print("ArUco 在 x 負方向且 y 正方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,1,-1,0,0,0,0,0,0,0,0,0))
                time.sleep(3)

            elif self.x < -100 and self.y < -100:
                print("ArUco 在 x 負方向且 y 負方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,1,1,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            elif self.x > 100 and self.y < -100:
                print("ArUco 在 x 正方向且 y 負方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,-1,1,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            elif self.x >100:
                print("ArUco 在 x 正方向且y 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,-1,0,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            elif self.x < -100:
                print("ArUco 在 x 負方向且y 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,1,0,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            elif self.y > 100:
                print("ArUco 在 y 正方向且x 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0,-1,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            elif self.y < -100:
                print("ArUco 在 y 負方向且x 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0,1,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
            
            else:
                print("drone 的位置在 ArUco 16 標準內")
                print("持續下降尋找 ArUco 0")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                    self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0,0,1,0,0,0,0,0,0,0,0))
                time.sleep(3)
                
        # 確認 ArUco 標記的類型
        elif self.id == self.id_to_find_0:  
            print("Find aruco marker 0")

            # 確認 ArUco 距離飛機距離
            if self.x > 30 and self.y > 30:
                print("ArUco 在 x 正方向且 y 正方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,-0.3,-0.3,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
                    
            elif self.x < -30 and self.y > 30:
                print("ArUco 在 x 負方向且 y 正方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0.3,-0.3,0,0,0,0,0,0,0,0,0))
                time.sleep(3)

            elif self.x < -30 and self.y < -30:
                print("ArUco 在 x 負方向且 y 負方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0.3,0.3,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
                    
            elif self.x > 30 and self.y < -30:
                print("ArUco 在 x 正方向且 y 負方向")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,-0.3,0.3,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
                    
            elif self.x >30:
                print("ArUco 在 x 正方向且y 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,-0.3,0,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
                    
            elif self.x < -30:
                print("ArUco 在 x 負方向且y 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0.3,0,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
                    
            elif self.y > 30:
                print("ArUco 在 y 正方向且x 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                            self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0,-0.3,0,0,0,0,0,0,0,0,0))
                time.sleep(3)
                    
            elif self.y < -30:
                print("ArUco 在 y 負方向且x 方向在標準內")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0,0.3,0,0,0,0,0,0,0,0,0))
                time.sleep(3)

            else:
                print("ArUco 位置在標準內")
                print("持續下降到地面")
                self.vehicle.mav.send(utility.mavlink.MAVLink_set_position_target_local_ned_message(10,self.vehicle.target_system,
                        self.vehicle.target_component,utility.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,int(0b110111111000) ,0,0,0.5,0,0,0,0,0,0,0,0))
                time.sleep(3)

    # 定義找不到 ArUco 的函數
    def not_found_aruco(self):

        if self.altitude <= 1.5:
            print("無人機高度低於", self.altitude, "m,找不到 ArUco")
            print("切換成 LAND 模式")
            self.mode_change_to_land()
            time.sleep(10)
            exit()
            
        else:
            print("找不到 ArUco")
    
    # 定義降落模式函數
    def mode_change_to_land(self):

        # desired flight mode
        FLIGHT_MODE = 'LAND'

        # get supported flight modes
        flight_modes = self.vehicle.mode_mapping()
    
        # check the desired flight mode is supported
        if FLIGHT_MODE not in flight_modes.keys():

            # inform user that desired mode is not supported by the vehicle
            print(FLIGHT_MODE, 'is not supported')

            # exit the code
            exit()
        
        # create change mode message
        set_mode_message = dialect.MAVLink_command_long_message(
            target_system=self.vehicle.target_system,
            target_component=self.vehicle.target_component,
            command=dialect.MAV_CMD_DO_SET_MODE,
            confirmation=0,
            param1=dialect.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            param2=flight_modes[FLIGHT_MODE],
            param3=0,
            param4=0,
            param5=0,
            param6=0,
            param7=0
        )
        # catch HEARTBEAT message
        message = self.vehicle.recv_match(type=dialect.MAVLink_heartbeat_message.msgname, blocking=True)

        # convert this message to dictionary
        message = message.to_dict()

        # get the mode id
        mode_id = message['custom_mode']

        # get mode name
        flight_mode_names = list(flight_modes.keys())
        flight_mode_ids = list(flight_modes.values())
        flight_mode_index = flight_mode_ids.index(mode_id)
        land_flight_mode_name = flight_mode_names[flight_mode_index ]

       # change flight mode
        self.vehicle.mav.send(set_mode_message)

        # do below always
        while True:

            # catch COMMAND_ACK message
            message = self.vehicle.recv_match(type=dialect.MAVLink_command_ack_message.msgname, blocking=True)

            # convert this message to dictionary
            message = message.to_dict()

            # check is the COMMAND_ACK is for DO_SET_MODE
            if message['command'] == dialect.MAV_CMD_DO_SET_MODE:

                # check the command is accepted or not
                if message['result'] == dialect.MAV_RESULT_ACCEPTED:

                    # inform the user
                    print("Changeing mode to", FLIGHT_MODE, "accepted from the vehicle")

                # not accepted
                else:

                    # inform the user
                    print("Changeing mode to", FLIGHT_MODE, "failed")

                # break the loop
                break
        
    # 定義導引模式函數
    def mode_change_to_guided(self):

        # desired flight mode
        FLIGHT_MODE = 'GUIDED'

        # get supported flight modes
        flight_modes = self.vehicle.mode_mapping()
    
        # check the desired flight mode is supported
        if FLIGHT_MODE not in flight_modes.keys():

            # inform user that desired mode is not supported by the vehicle
            print(FLIGHT_MODE, 'is not supported')

            # exit the code
            exit()
        
        # create change mode message
        set_mode_message = dialect.MAVLink_command_long_message(
            target_system=self.vehicle.target_system,
            target_component=self.vehicle.target_component,
            command=dialect.MAV_CMD_DO_SET_MODE,
            confirmation=0,
            param1=dialect.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            param2=flight_modes[FLIGHT_MODE],
            param3=0,
            param4=0,
            param5=0,
            param6=0,
            param7=0
        )
        # catch HEARTBEAT message
        message = self.vehicle.recv_match(type=dialect.MAVLink_heartbeat_message.msgname, blocking=True)

        # convert this message to dictionary
        message = message.to_dict()

        # get the mode id
        mode_id = message['custom_mode']

        # get mode name
        flight_mode_names = list(flight_modes.keys())
        flight_mode_ids = list(flight_modes.values())
        flight_mode_index = flight_mode_ids.index(mode_id)
        flight_mode_name = flight_mode_names[flight_mode_index ]

       # change flight mode
        self.vehicle.mav.send(set_mode_message)

        # do below always
        while True:

            # catch COMMAND_ACK message
            message = self.vehicle.recv_match(type=dialect.MAVLink_command_ack_message.msgname, blocking=True)

            # convert this message to dictionary
            message = message.to_dict()

            # check is the COMMAND_ACK is for DO_SET_MODE
            if message['command'] == dialect.MAV_CMD_DO_SET_MODE:

                # check the command is accepted or not
                if message['result'] == dialect.MAV_RESULT_ACCEPTED:

                    # inform the user
                    print("Changeing mode to", FLIGHT_MODE, "accepted from the vehicle")

                # not accepted
                else:

                    # inform the user
                    print("Changeing mode to", FLIGHT_MODE, "failed")

                # break the loop
                break


        # catch HEARTBEAT message
        message = self.vehicle.recv_match(type=dialect.MAVLink_heartbeat_message.msgname, blocking=True)

        # convert this message to dictionary
        message = message.to_dict()

        # get the mode id
        mode_id = message['custom_mode']

        # get mode name
        flight_mode_names = list(flight_modes.keys())
        flight_mode_ids = list(flight_modes.values())
        flight_mode_index = flight_mode_ids.index(mode_id)
        self.flight_mode_name = flight_mode_names[flight_mode_index ]

        #print heartbeat message
        print("Now mode is:", self.flight_mode_name)

        # change flight mode
        self.vehicle.mav.send(set_mode_message)

    # 取得航點資訊
    def get_waypoint_info(self):

        # 发送请求命令，请求 MISSION_CURRENT 消息
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,          # target_system
            self.vehicle.target_component,       # target_component
            utility.mavlink.MAV_CMD_REQUEST_MESSAGE,  # command
            0,                              # confirmation
            utility.mavlink.MAVLINK_MSG_ID_MISSION_CURRENT,  # param1: 消息ID
            0, 0, 0, 0, 0, 0                # params 2-7: 保留参数
        )

        # 等待并接收 MISSION_CURRENT 消息
        while True:
            time.sleep(3)
            message = self.vehicle.recv_match(type='MISSION_CURRENT', blocking=True)
            if message:
                if self.flight_mode_name == "AUTO":
                    wp_point = {message.seq}
                    # 將集合轉換成整數
                    self.wp_point = list(wp_point)[0]
                    print(f"當前任務編號: {self.wp_point}")
                    print(f"當前任務總數: {self.wp_point_count}")
                    break

                else:
                    wp_point = {message.seq}
                    # 將集合轉換成整數
                    self.wp_point = list(wp_point)[0]
                    break
    
    # 取得航點總數
    def get_waypoint_count(self):

        # create mission request list message
        message = dialect.MAVLink_mission_request_list_message(target_system=self.vehicle.target_system,
                                                            target_component=self.vehicle.target_component,
                                                            mission_type=dialect.MAV_MISSION_TYPE_MISSION)

        # send the message to the vehicle
        self.vehicle.mav.send(message)

        # wait mission count message
        message = self.vehicle.recv_match(type=dialect.MAVLink_mission_count_message.msgname,
                                    blocking = True)

        # convert this message to dictionary
        message = message.to_dict()

        # get the mission item count
        count = message["count"]
        print("Total mission item count:", count)

        # create mission item list
        mission_item_list = []

        # gat the mission items
        for i in range(count):
            message = dialect.MAVLink_mission_request_int_message(target_system=self.vehicle.target_system,
                                                                target_component=self.vehicle.target_component,
                                                                seq=i,
                                                                mission_type=dialect.MAV_MISSION_TYPE_MISSION)
            
            # send message request int message to the vehicle
            self.vehicle.mav.send(message)

            # wait mission count message
            message = self.vehicle.recv_match(type=dialect.MAVLink_mission_item_int_message.msgname,
                                        blocking = True) 
            
            # convert this message to dictionary
            message = message.to_dict()

            # add mission items to the list
            mission_item_list.append(message)

        for mission_item in mission_item_list:
            print("Seq", mission_item['seq'])
            self.wp_point_count = mission_item['seq'] 

    # 清除原有任務
    def clear_mission(self):

        # 发送清除所有任务的命令
        self.vehicle.mav.mission_clear_all_send(
            self.vehicle.target_system,
            self.vehicle.target_component
        )
        print("All missions cleared")
    
    # 設定 WP_YAW_BEHAVIOR 参数
    def set_wp_yaw_behavior(self,value):
        # WP_YAW_BEHAVIOR 参数索引为 4（根据参数索引表，可以在Mission Planner中找到）
        param_index = 4
        self.vehicle.mav.param_set_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            b'WP_YAW_BEHAVIOR',
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_INT8
        )

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()