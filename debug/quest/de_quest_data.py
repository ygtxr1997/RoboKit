import time
from robokit.network.vr_handler import OculusReader, QuestHandler


# # Op1. OculusReader
    # oculus_reader = OculusReader()
    #
    # while True:
    #     time.sleep(0.3)
    #     print(oculus_reader.get_transformations_and_buttons())

# Op2. QuestHandler
quest_handler = QuestHandler()
while True:
    time.sleep(0.1)
    print(quest_handler.get_latest_euler())