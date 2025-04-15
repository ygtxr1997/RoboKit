import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d


# cur = [0, 0, 90]
# base_cmd = [0, 180, 45]
# local_cmd = [180, 0, -45]
# nxt = [180, 180, -45]
#
# cur = R.from_euler('XYZ', cur, degrees=True)
# base = R.from_euler('XYZ', base_cmd, degrees=True)
# local = R.from_euler('xyz', local_cmd, degrees=True)
# nxt = R.from_euler('XYZ', nxt, degrees=True)
#
# def qprint(prefix: str, q):
#     out_str = str(prefix) + " "
#     for v in q.as_quat():
#         out_str += f"{v:.4f}\t"  # (x,y,z,w)
#     print(out_str, q.as_euler('xyz', degrees=True))
#
# qprint('cur:', cur)
# qprint('base:', base)
# qprint('local:', local)
# qprint('nxt:', nxt)
#
# # 四元数乘法
# q_mul = base * cur * base.inv() # 或者使用 q1_quat @ q2_quat
# qprint("nxt =", q_mul)  # 显示乘积的四元数
#
# # 四元数除法 (q1 / q2 = q1 * q2^-1)
# q_div = nxt * cur.inv()  # q1 除以 q2
# qprint("base =", q_div)  # 显示除法结果
#
# # 四元数除法 (q1 / q2 = q1 * q2^-1)
# q_div = cur.inv() * base * cur  # q1 除以 q2
# qprint("local =", q_div)  # 显示除法结果


cur = [0, 0, 90]
base_cmd = [0, 180, 45]
local_cmd = [180, 0, -45]
nxt = [180, 180, -45]

cur_rad = np.radians(cur)
cur_R = transforms3d.euler.euler2mat(*cur_rad, axes='sxyz')
cur_q = transforms3d.euler.euler2quat(*cur_rad, axes='sxyz')
cur_q_inv = transforms3d.quaternions.qinverse(cur_q)

base_rad = np.radians(base_cmd)
base_R = transforms3d.euler.euler2mat(*base_rad, axes='sxyz')
base_q = transforms3d.euler.euler2quat(*base_rad, axes='sxyz')

local_rad = np.radians(local_cmd)
local_q = transforms3d.euler.euler2quat(*local_rad, axes='rxyz')
print('local_q:', local_q)

nxt_rad = np.radians(nxt)
nxt_q = transforms3d.euler.euler2quat(*local_rad, axes='sxyz')
print('nxt_q:', nxt_q)
print('base_q:', base_q)

print("==" * 20, "Predicted", "==" * 20)
local_q = transforms3d.quaternions.qmult(transforms3d.quaternions.qmult(cur_q_inv, base_q), cur_q)
print('local_q=', local_q)
local = transforms3d.euler.quat2euler(local_q, axes='rxyz')
local = [angle * 180 / 3.141592653589793 for angle in local]
print('local=', local)

nxt_q = transforms3d.quaternions.qmult(local_q, cur_q)
print('nxt_q=', nxt_q)

base_q = transforms3d.quaternions.qmult(transforms3d.quaternions.qmult(cur_q, local_q), cur_q_inv)
print('base_q=', base_q)

