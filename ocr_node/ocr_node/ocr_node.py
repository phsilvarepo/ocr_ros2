import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ocr_node.msg import OCRResult

# Optional: PaddleOCR
from paddleocr import PaddleOCR


class OCRNode(Node):

    def __init__(self):
        super().__init__('ocr_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            OCRResult,
            '/ocr_result',
            10
        )

        self.bridge = CvBridge()
        self.ocr = PaddleOCR(use_angle_cls=True)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.ocr.ocr(frame)

        for line in results:
            for box, (text, confidence) in line:

                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]

                bbox = [
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords))
                ]

                msg_out = OCRResult()
                msg_out.text = text
                msg_out.confidence = float(confidence)
                msg_out.bbox = bbox

                self.publisher.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = OCRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
