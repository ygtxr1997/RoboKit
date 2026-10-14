import roslibpy
import traceback
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
# saving_root = "/home/geyuan/local_soft/TCL/1201_screw_bulb_turn_off_table/"
# task_instruction = "Screw the bulb into the light socket, and turn off the light after the bulb is on. [ENV:table]"
# saving_root = "/home/geyuan/local_soft/TCL/1201_screw_bulb_turn_off_object/"
# task_instruction = "Screw the bulb into the light socket, and turn off the light after the bulb is on. [ENV:object]"
# saving_root = "/home/geyuan/local_soft/TCL/1201_screw_bulb_turn_off_light/"
# task_instruction = "Screw the bulb into the light socket, and turn off the light after the bulb is on. [ENV:light]"


''' Tower '''
# saving_root = "/home/geyuan/local_soft/TCL/0209_tower_boby/"
# saving_root = "/home/geyuan/local_soft/TCL/0209_tower_boby_hard/"
# saving_root = "/home/geyuan/local_soft/TCL/0209_tower_boby_easy/"
# task_instruction = "Pick up the purple disc from the pole and place it on the sponge."
# task_instruction = "Pick up the purple disc from the sponge and insert it onto the pole."

''' Mouse '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_put_mouse/"
# task_instruction = "Pick up the mouse from the wireless charger and place it onto the black mouse pad."
saving_root = "/home/geyuan/local_soft/TCL/0417_put_mouse_hard/"
task_instruction = "Pick up the mouse from the black mouse pad and place it onto the wireless charger."

''' Ethernet cable '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_ethernet/"
# task_instruction = "Plug the ethernet cable into the switch port."
# saving_root = "/home/geyuan/local_soft/TCL/260514_ethernet/"
# task_instruction = "Disconnect the ethernet cable from the `uplink` port, then insert it into port `1`."

''' GreenYellowRed '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_greenyellowred/"
# task_instruction = "Sequentially take off the three wooden blocks (green, yellow, and red) from the rod and place them on the black mouse pad."

''' Test Tube '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_test_tube/"
# task_instruction = "Pull out the test tube from the test tube rack"

''' Test Tube '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_french_press/"
# task_instruction = "Lift the lid off the French press and set it on the towel."

''' Brush '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_brush/"
# task_instruction = "Disassemble the brush and the dustpan"

''' Plastic Egg '''
# saving_root = "/home/geyuan/local_soft/TCL/0417_plastic_egg/"
# task_instruction = "Pick the plastic egg and place it into the egg box."

''' Debug '''
# saving_root = "/home/geyuan/local_soft/TCL/tmp/"
# task_instruction = "Warning! this is a debug task!"


''' disk force 3 '''
# saving_root = "/home/geyuan/local_soft/TCL/0731_disk_force_9/"
# task_instruction = "Pick up the purple disc from the sponge and insert it onto the pole."

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
    print("Error: {}".format(e))
    traceback.print_exc()
except KeyboardInterrupt:
    print("用户按下 Ctrl+C，程序中断。")
finally:
    print("Please wait for all resources to be released...")
    controller.stop()

print("OK!")