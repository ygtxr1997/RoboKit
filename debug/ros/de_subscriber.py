import roslibpy
import roslibpy.actionlib
from tqdm import tqdm
import time

client = roslibpy.Ros(host='192.168.5.242', port=9090) # Change host to the IP of the robot
client.run()


my_topic = "/end_pose"  # /joint_states_single, /joint_ctrl_single, /end_pose
topic_type = client.get_topic_type(topic=my_topic)
print(topic_type)
message_info = client.get_message_details(message_type=topic_type)
print(message_info)

def on_message(message):
    print(message)

subscriber = roslibpy.Topic(client, my_topic,
                            topic_type)
subscriber.subscribe(on_message)


try:
    while True:
        time.sleep(0.9)
except KeyboardInterrupt:
    print("Exiting...")

subscriber.unsubscribe()
