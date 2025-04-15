from geometry_msgs.msg import PoseStamped

def convert_to_pose_array(path, frame_id="base_link"):
    poses = []
    for (x, y, z) in path:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0  # Identity quaternion
        poses.append(pose)
    return poses
