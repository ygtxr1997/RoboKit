import roslibpy

from robokit.robots.robot_client_inovo import RobotClient
from robokit.controllers.tele_control import PS5DualSenseIMUController


robot_ip = '192.168.1.7'
client = roslibpy.Ros(host=robot_ip, port=9090) # Change host to the IP of the robot
client.run()

# Sanity check to see if we are connected
print('Verifying the ROS target is connected?', client.is_connected)

env_source_params = {
    "enable_auto_ae_wb": True,
}

env_light_params = {
    "enable_auto_ae_wb": False,
    "ae_wb_params": [{
        "camera_idx": -1,
        "exposure": 30,
    }],
}

### Params for data_manager collector ###

''' debug '''
# saving_root = "/home/geyuan/local_soft/TCL/1201_debug/"
# task_instruction = "Debug task."

''' write the word `AI` on the whiteboard '''
# saving_root = "/home/geyuan/local_soft/TCL/1201_write_AI_on_whiteboard/"
# task_instruction = "Use the pen to write the word 'AI' on the whiteboard."

''' wipe the writing on the blackboard '''
# saving_root = "/home/geyuan/local_soft/TCL/1201_wipe_blackboard/"
# task_instruction = "Wipe the writing on the blackboard."

''' pick and place wooden eggs '''
# saving_root = "/home/geyuan/local_soft/TCL/1024_eggs_pick_place/"
# task_instruction = "Pick up two eggs from a plate, and then place the eggs into a egg carton."

''' Sweep 20beans '''
# saving_root = "/home/geyuan/local_soft/TCL/1024_sweep_bean/"
# task_instruction = "Pick up the broom and sweep the coffee beans on the table into the dustpan."

''' Pour water '''
# saving_root = "/home/geyuan/local_soft/TCL/1024_pour_water/"
# saving_root = "/home/geyuan/local_soft/TCL/1201_pour_water/"
# task_instruction = "Pick up the measuring cup and pour the water into a glass."

''' Wipe '''
# saving_root = "/home/geyuan/local_soft/TCL/1024_wipe_white_board/"
# task_instruction = "Wiping the writing on the whiteboard."

''' Banana '''
# saving_root = "/home/geyuan/local_soft/TCL/1201_banana/"
# task_instruction = "pick up the banana"

''' Insert Red Pepper'''
# saving_root = "/home/geyuan/local_soft/TCL/1117_red_peper_pick_place/"
# task_instruction = "Pick up the red pepper on the box, and then place it into a cup."

''' Take carrot out to the Bowl '''
# saving_root = "/home/geyuan/local_soft/TCL/1117_carrot_pick_place"
# task_instruction = "Pick up the carrot from a plate, and then place it into a bowl."

''' Pot '''
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_light_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_object_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_table_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0627_pot_distractor_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/1201_pot/"
# task_instruction = "put the egg into the pot, then move the pot onto the stove"

''' Pepper '''
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_source/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_light_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_object_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_table_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0704_pepper_distractor_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/1201_pepper/"
# task_instruction = "orient the cup upright on its base and insert the chili pepper vertically into it."
''' Coffee '''
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_source/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_light_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_object_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_table_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/0709_coffee_distractor_rand/"
# saving_root = "/home/geyuan/local_soft/TCL/1201_coffee/"
# task_instruction = "use a spoon to scoop one spoonful of coffee beans from the source cup, then pour the beans into the target cup."

''' Bulb '''
# saving_root = "/home/geyuan/local_soft/TCL/1201_screw_bulb_table/"
# task_instruction = "Pick up the bulb, and screw it into the light socket."
# saving_root = "/home/geyuan/local_soft/TCL/1201_screw_bulb/"
# task_instruction = "Screw the bulb into the light socket, and turn on the light."
# saving_root = "/home/geyuan/local_soft/TCL/1201_screw_bulb_turn_off/"
# task_instruction = "Screw the bulb into the light socket, and turn off the light after the bulb is on."


''' Tower '''
saving_root = "/home/geyuan/local_soft/TCL/0209_tower_boby/"
task_instruction = "Pick up the purple disc from the pole and place it on the sponge."

env_params = env_source_params

rc = RobotClient(client)
controller = PS5DualSenseIMUController(
    robot=rc,
    saving_root=saving_root,
    task_instruction=task_instruction,
    saving_workers=6,
    action_fps=30, #15, #30,  # ori:30, rand:60
    **env_params
)

try:
    controller.start()
except Exception as e:
    print(e)
except KeyboardInterrupt:
    print("用户按下 Ctrl+C，程序中断。")
finally:
    controller.stop()

print("OK!")