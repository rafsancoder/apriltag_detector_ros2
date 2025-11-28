import cv2
from pupil_apriltags import Detector

import rclpy
from rclpy.node import Node


class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__("apriltag_detector")

        # === Open stream from rpicam-vid on the same Pi ===
        # rpicam-vid command on Pi:
        # rpicam-vid -t 0 --width 640 --height 480 --framerate 15 \
        #   --codec mjpeg --listen -o tcp://0.0.0.0:9000
        self.stream_url = "tcp://127.0.0.1:9000"
        self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            self.get_logger().error(
                f"Could not open video stream: {self.stream_url}"
            )
            raise RuntimeError("Cannot open video stream")

        # === AprilTag detector configuration ===
        self.detector = Detector(
            families="tag36h11",  # make sure printed tag is from this family
            nthreads=4,
            quad_decimate=1.0,    # >1 = faster, <1 = more accurate
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        # Process at ~15 Hz
        self.timer = self.create_timer(1.0 / 15.0, self.timer_callback)
        self.get_logger().info(
            "AprilTagDetectorNode started. Showing OpenCV window. Press ESC in the window to quit."
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("No frame received from camera")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        if tags:
            ids = [t.tag_id for t in tags]
            self.get_logger().info(f"Detected {len(tags)} tag(s): {ids}")

        # Draw detections
        for tag in tags:
            corners = tag.corners.astype(int)

            # Draw green box around tag
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Show tag ID
            cv2.putText(
                frame,
                f"ID {tag.tag_id}",
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # Show number of tags in top-left
        cv2.putText(
            frame,
            f"tags: {len(tags)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("AprilTag detection (ROS 2 Jazzy)", frame)

        # ESC key to quit
        if cv2.waitKey(1) & 0xFF == 27:
            self.get_logger().info("ESC pressed, shutting down node.")
            rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AprilTagDetectorNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Exception in AprilTagDetectorNode: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
