import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from paddleocr import PaddleOCR
import paddle  # <--- NEED THIS FOR THE GPU CHECK
import os
import cv2

class OCRNode(Node):
    def __init__(self):
        super().__init__('ocr_node')
        
        # Pulling settings from environment variables
        env_input = os.environ.get('INPUT_TOPIC', '/rgb')
        env_det_output = os.environ.get('OUTPUT_TOPIC_BB', '/paddle_ocr/detections')
        env_img_output = os.environ.get('OUTPUT_TOPIC_IMAGE', '/paddle_ocr/image')
        raw_conf = os.environ.get('CONFIDENCE_THRESHOLD', '0.6')
        
        self.conf_threshold = float(raw_conf)
        self.bridge = CvBridge()
        
        # Subscribers and Publishers
        self.subscription = self.create_subscription(Image, env_input, self.image_callback, 10)
        self.det_publisher = self.create_publisher(Detection2DArray, env_det_output, 10)
        self.img_publisher = self.create_publisher(Image, env_img_output, 10)

        # Initialize PaddleOCR
        # note: use_gpu=True will only work if your Docker has --gpus all
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)
        
        # LOGGING GPU STATUS
        cuda_available = paddle.is_compiled_with_cuda()
        device_place = paddle.get_device()
        self.get_logger().info(f"Using GPU (compiled with CUDA): {cuda_available}")
        self.get_logger().info(f"Paddle is running on: {device_place}")
        self.get_logger().info(f"OCR Node started. Listening: {env_input}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.ocr.ocr(frame, cls=True)
        
        det_array_msg = Detection2DArray()
        det_array_msg.header = msg.header 

        if results and results[0] is not None:
            for line in results[0]:
                if line is None: continue
                
                box, (text, confidence) = line
                if float(confidence) < self.conf_threshold: continue

                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                det_msg = Detection2D()
                det_msg.header = msg.header
                det_msg.bbox.center.position.x = float(min_x + (max_x - min_x) / 2.0)
                det_msg.bbox.center.position.y = float(min_y + (max_y - min_y) / 2.0)
                det_msg.bbox.size_x = float(max_x - min_x)
                det_msg.bbox.size_y = float(max_y - min_y)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(text) 
                hyp.hypothesis.score = float(confidence)
                det_msg.results.append(hyp)
                det_array_msg.detections.append(det_msg)

                cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
                cv2.putText(frame, f"{text} ({confidence:.2f})", (int(min_x), int(min_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.det_publisher.publish(det_array_msg)
        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        annotated_msg.header = msg.header
        self.img_publisher.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OCRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
