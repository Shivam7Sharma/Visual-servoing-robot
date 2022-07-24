#!/usr/bin/env python3
# license removed for brevity
#!/usr/bin/env python3
# license removed for brevity
import math
import random
import rospy
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Float64


def talker():		
		# pub_q1_pos, pub_q2_pos - publish the joint position to the joints
		pub_q1_pos = rospy.Publisher('/vbmbot/joint1_position_controller/command', Float64, queue_size=10)
		pub_q2_pos = rospy.Publisher('/vbmbot/joint2_position_controller/command', Float64, queue_size=10)
		# pub_q1_vel, pub_q2_vel - publish the joint velocity to the joints
		pub_q1_vel = rospy.Publisher('/vbmbot/joint1_velocity_controller/command', Float64, queue_size=10)
		pub_q2_vel = rospy.Publisher('/vbmbot/joint2_velocity_controller/command', Float64, queue_size=10)
		#initialize the node
		#at node initialization, the position controller is active 
		# so only position(joint angles) commands can be given to joints
		rospy.init_node('joint_manip_talker', anonymous=True)
		rate = rospy.Rate(1)# meaning 1 message published in 1 sec
		rospy.sleep(5) 
		random.seed()
		#target position given to move the joints away from home position     
		# q1_pos = 2.09 
		# q2_pos = 1.57 
		q1_pos = 0.52
		q2_pos = -0.52
		pub_q1_pos.publish(q1_pos)
		pub_q2_pos.publish(q2_pos)
		rospy.sleep(5)
		#once the joints have moved from home position, 
		# the position controller is stopped and velocity controller is started. 
		# We use ros inbuilt switch_controller service for that.
		rospy.wait_for_service('/vbmbot/controller_manager/switch_controller')
		try:
			sc_service = rospy.ServiceProxy('/vbmbot/controller_manager/switch_controller', SwitchController)
			start_controllers = ['joint1_velocity_controller','joint2_velocity_controller']
			stop_controllers = ['joint1_position_controller','joint2_position_controller']
			strictness = 2
			start_asap = False
			timeout = 0.0
			res = sc_service(start_controllers,stop_controllers, strictness, start_asap,timeout)

		except rospy.ServiceException as e:
			print("Service Call Failed")	
		
		# while not rospy.is_shutdown():				
		# 	# once the controller switch is successful the joints will 
		# 	# receive random velocity from the velocity controller 
		# 	# and keep moving until ROS is shutdown
		# 	q1_vel = random.uniform(-0.5, 0.5)
		# 	q2_vel = random.uniform(-0.5, 0.5)

		# 	pub_q1_vel.publish(q1_vel)
		# 	pub_q2_vel.publish(q2_vel)

		# 	rate.sleep()		


if __name__ == '__main__':	
	try:
		talker()
		rospy.sleep(30)
	except rospy.ROSInterruptException:
		pass
