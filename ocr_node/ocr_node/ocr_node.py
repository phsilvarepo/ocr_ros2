import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ocr_interfaces.msg import OCRResult, OCRResultArray 
from paddleocr import PaddleOCR
import os

class OCRNode(Node):
    def __init__(self):
        super().__init__('ocr_node')
        
        # Pulling settings from environment variables
        env_input = os.environ.get('INPUT_TOPIC', '/rgb')
        env_output = os.environ.get('OUTPUT_TOPIC', '/ocr_detection')
        raw_conf = os.environ.get('CONFIDENCE_THRESHOLD', '0.6')
        
        try:
            self.conf_threshold = float(raw_conf)
        except ValueError:
            self.get_logger().warn(f"Invalid CONFIDENCE_THRESHOLD '{raw_conf}'. Defaulting to 0.6")
            self.conf_threshold = 0.6
        
        self.subscription = self.create_subscription(
            Image, 
            env_input, 
            self.image_callback, 
            10
        )

        self.publisher = self.create_publisher(
            OCRResultArray, 
            env_output, 
            10
        )

        self.bridge = CvBridge()
        
        # Initialize PaddleOCR
        # note: drop_score is Paddle's internal filter for the detection boxes
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        self.get_logger().info(
            f"OCR Node started.\n"
            f"Listening: {env_input}\n"
            f"Threshold: {self.conf_threshold}"
        )

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run Inference
        results = self.ocr.ocr(frame, cls=True)
        
        # Create the array container
        array_msg = OCRResultArray()
        array_msg.header = msg.header 

        if results and results[0] is not None:
            for line in results[0]:
                if line is None: 
                    continue
                
                # Extract data from PaddleOCR output structure
                box, (text, confidence) = line
                
                # --- CONFIDENCE THRESHOLD IMPLEMENTATION ---
                if float(confidence) < self.conf_threshold:
                    continue
                
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                
                single_result = OCRResult()
                single_result.text = text
                single_result.confidence = float(confidence)
                single_result.bbox = [
                    int(min(x_coords)), int(min(y_coords)),
                    int(max(x_coords)), int(max(y_coords))
                ]
                
                array_msg.results.append(single_result)

        # Publish once per frame
        self.publisher.publish(array_msg)

# --- THE MAIN ENTRY POINT ---
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
