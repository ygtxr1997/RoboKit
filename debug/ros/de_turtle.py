import os
import time

import roslibpy
import pygame ## A library used to read joystick inputs


class TurtleClient:
    pass


def main():
    robot_ip = '127.0.0.1'
    client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
    client.run()

    print('Verifying the ROS target is connected?', client.is_connected)    # HW - If its not connected 

    topics = client.get_topics()
    print(topics)
    topic_type = client.get_topic_type(topic=topics[2])
    print(topic_type)
    message_info = client.get_message_details(message_type=topic_type)
    print(message_info)

    """
    In CMD:
    rostopic pub /turtle1/cmd_vel geometry_msgs/Twist -r 1 -- '[2.0, 0.0, 0.0]' '[0.0, 0.0, 1.8]'
    """
    publisher = roslibpy.Topic(
        client, '/turtle1/cmd_vel', 'geometry_msgs/Twist'
    )
    while True:
        publisher.publish(
            roslibpy.Message({
                "linear": {'x': 2.0, 'y': 0.0, 'z': 0.0},
                "angular": {'x': 0.0, 'y': 0.0, 'z': 1.8}
            })
        )
        time.sleep(0.1)


if __name__ == "__main__":
    main()
