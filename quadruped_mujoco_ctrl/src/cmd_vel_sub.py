#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_subscriber')
        self.cmd_linear_x = 0.0
        self.cmd_angular_z = 0.0
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.get_logger().info('CmdVelSubscriber node has been started.')

    def cmd_vel_callback(self, msg):
        cmd_linear_x = msg.linear.x
        cmd_angular_z = msg.angular.z
        self.get_logger().info(f'Received cmd_vel: linear.x={cmd_linear_x}, angular.z={cmd_angular_z}')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = CmdVelSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        
