import matplotlib.pyplot as plt
from nerfloc.utils.base import get_names_qctcs
from nerfloc.utils.plot import plot_cam_poses
from nerfloc.utils.read_write_model import qvec2rotmat

#! Input poses file
poses_file = '/home/sunycs/rashik/dataset/dtu_scene6/poses.txt'

#! Read poses file
names, poses = get_names_qctcs(poses_file)

#! Quarternion to R conversion
R0 = qvec2rotmat(poses[0].numpy()) 
print(poses[0])
print("=")
print(R0)

#! Setup 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
ax.view_init(elev=-70, azim=-90, roll=0)

#! Plot Cam Poses
plot_cam_poses(poses.numpy(), ax)

#! Save Plot
plt.savefig(f"poses.jpg")
plt.close() 