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
        
        # Pulling topics from environment variables for Docker flexibility
        env_input = os.environ.get('INPUT_TOPIC', '/rgb')
        env_output = os.environ.get('OUTPUT_TOPIC', '/ocr_detection')
        
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
        # Initialize PaddleOCR (This will download models on first run)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        self.get_logger().info(f"OCR Node started. Listening on {env_input}...")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run Inference
        results = self.ocr.ocr(frame, cls=True)
        
        # Create the array container
        array_msg = OCRResultArray()
        array_msg.header = msg.header # Essential for syncing with the original image

        if results and results[0] is not None:
            for line in results[0]:
                if line is None: continue
                
                box, (text, confidence) = line
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
    # 1. Initialize rclpy
    rclpy.init(args=args)
    
    # 2. Instantiate the node
    node = OCRNode()
    
    try:
        # 3. Spin the node so it processes callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 4. Shutdown cleanly
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
