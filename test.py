#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from action_msgs.msg import GoalStatusArray
from action_msgs.msg import GoalStatus

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np


class ObjectMonitor(Node):
    def __init__(self):
        super().__init__('object_monitor')

        self.count = 0
        self.seen = False

        # 가려짐 판단 기준
        self.dark_th = 40     # 평균 밝기
        self.var_th = 15      # 분산

        # QoS 설정
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.img_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.img_cb,
            image_qos
        )

        self.nav_sub = self.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self.nav_cb,
            nav_qos
        )

        self.get_logger().info('ObjectMonitor node started (CompressedImage + QoS)')

    # 이미지 callback
    def img_cb(self, msg: CompressedImage):
        # CompressedImage → OpenCV
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        mean_val = np.mean(gray)
        var_val = np.var(gray)

        # 카메라 가려짐 판단
        occluded = (mean_val < self.dark_th) or (var_val < self.var_th)

        # "가려짐이 새로 발생한 순간"만 카운트
        if occluded and not self.seen:
            self.count += 1
            self.get_logger().warn(
                f'Camera occluded! Count={self.count}, mean={mean_val:.1f}, var={var_val:.1f}'
            )

        self.seen = occluded

    # Nav2 status callback
    def nav_cb(self, msg: GoalStatusArray):
        for status in msg.status_list:
            if status.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f'Goal reached! Total occlusions: {self.count}'
                )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

