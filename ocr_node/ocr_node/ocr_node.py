import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ocr_interfaces.msg import OCRResult
from paddleocr import PaddleOCR
import os

class OCRNode(Node):

    def __init__(self):
        super().__init__('ocr_node')
        
        env_input = os.environ.get('INPUT_TOPIC', '/rgb')
        env_output = os.environ.get('OUTPUT_TOPIC', '/ocr_detection')
        
        self.declare_parameter('image_topic', env_input)
        self.declare_parameter('output_topic', env_output)
        
        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            OCRResult,
            self.output_topic,
            10
        )

        self.bridge = CvBridge()
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            ocr_version='PP-OCRv4',   # avoid PP-OCRv5 server models
            lang='en'
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.ocr.ocr(frame, cls=True)
        
        if not results or results[0] is None:
            return
        
        for line in results[0]:
            if line is None:
                continue
            box, (text, confidence) = line
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            bbox = [
                int(min(x_coords)), int(min(y_coords)),
                int(max(x_coords)), int(max(y_coords))
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
