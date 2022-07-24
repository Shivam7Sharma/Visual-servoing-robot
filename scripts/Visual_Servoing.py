#!/usr/bin/env python3
import rospy
import sys
import cv2
import math
import csv
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState


count = 0 
j_ang = []
error = 1
init_pose = []
final_pose = []
feat = []

pub_q1_vel = rospy.Publisher('/vbmbot/joint1_velocity_controller/command', Float64, queue_size=10)
pub_q2_vel = rospy.Publisher('/vbmbot/joint2_velocity_controller/command', Float64, queue_size=10)

def hough_circles(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,2,45,param1=40,param2=20,minRadius=0,maxRadius=15)

    circles = np.uint16(np.around(circles))
    features = []

    for i in circles[0,:]:
        # draw the outer circle
        print("Centre of the Hough circles is:",i[0],i[1])
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

        features.append((i[0],i[1]))

    cv2.imshow('detected circles',img)
    cv2.waitKey(0)

def cent(img,mask):
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img,(cX,cY),2,(255,255,255),-1)
    return img,(cX,cY)

def feature_extract(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    green_lower = np.array([58,10,95])
    green_upper = np.array([78,255,200])
    purple_lower = np.array([138,20,85])
    purple_upper = np.array([158,255,170])
    red_lower = np.array([118,200,75])
    red_upper = np.array([128,255,200])
    blue_lower = np.array([0,50,45])
    blue_upper = np.array([10,255,250])
    blue1_lower = np.array([175,50,45])
    blue1_upper = np.array([180,255,250])
    blue_mask1 = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_mask2 = cv2.inRange(hsv, blue1_lower, blue1_upper)
    blue_mask = blue_mask1+blue_mask2
    blue_res = cv2.bitwise_and(img,img, mask= blue_mask)
    blue_res,blue_cen = cent(img,blue_mask)
    purp_mask = cv2.inRange(hsv, purple_lower, purple_upper)
    purp_res = cv2.bitwise_and(img,img, mask= purp_mask)
    purp_res,purp_cen = cent(img,purp_mask)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_res = cv2.bitwise_and(img,img, mask= green_mask)
    green_res,green_cen = cent(img,green_mask)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    red_res = cv2.bitwise_and(img,img, mask= red_mask)
    red_res,red_cen = cent(img,red_mask)
    print("The centre for Red Circle is at: ",red_cen)
    print("The centre for Purple Circle is at: ",purp_cen)
    print("The centre for Green Circle is at: ",green_cen)
    print("The centre for Blue Circle is at: ",blue_cen)
    cv2.imshow("win",img)
    cv2.waitKey(3)
    return img,[blue_cen,purp_cen,red_cen,green_cen]

def jointstate_callback(msg):
    global j_ang
    j_ang = msg.position

def inv_jacobian(q1,q2,Vc):
    a1 = np.array([[math.cos(q1), -math.sin(q1),0,0.5*math.cos(q1)],[math.sin(q1),math.cos(q1),0,0.5*math.sin(q1)],[0,0,1,0],[0,0,0,1]])
    a2 = np.array([[math.cos(q2), -math.sin(q2),0,0.5*math.cos(q2)],[math.sin(q2),math.cos(q2),0,0.5*math.sin(q2)],[0,0,1,0],[0,0,0,1]])
    T  = np.matmul(a1,a2)
    Oc = T[0:-1,[-1]]
    O2 = a1[0:-1,[-1]]
    zi = np.array([[0],[0],[1]])
    s1 = Oc
    s2 = Oc-O2
    skew_s1 = np.array([[0,-1*s1[-1,0],s1[1,0]],[s1[-1,0],0,-1*s1[0,0]],[-1*s1[1,0],s1[0,0],0]])
    skew_s2 = np.array([[0,-1*s2[-1,0],s2[1,0]],[s2[-1,0],0,-1*s2[0,0]],[-1*s2[1,0],s2[0,0],0]])
    skew_s1.transpose()
    skew_s2.transpose()
    Jv1 = np.matmul(skew_s1,zi)
    Jv2 = np.matmul(skew_s2,zi)
    # j1  = np.concatenate((Jv1,zi),axis=1)
    Jacobian = np.concatenate((np.concatenate((Jv1,zi),axis=0),np.concatenate((Jv2,zi),axis=0)),axis = 1)
    Inv_Jacobian = np.linalg.pinv(Jacobian)
    # print("jacobian:",Inv_Jacobian)
    Joint_Vels = np.matmul(Inv_Jacobian,Vc)
    print("Joint Angles:",Joint_Vels)
    return Joint_Vels[0,0],Joint_Vels[1,0]


class image_converter:


    def __init__(self):
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/vbmbot/camera1/image_raw",Image,self.callback)
    
        
    def callback(self,data):
        global count
        global init_pose
        global final_pose
        global error
        global j_ang
        global feat
        
        
        count +=1 
        print(count)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        if count == 11:
            img = cv_image.copy()
            print("Received an image! Please wait! Implementing Image Processing.")
            rospy.sleep(1)      
            img_with_detection,init_pose = feature_extract(img)
            cv2.imwrite("/home/pinak/catkin_ws/src/img.jpg",img_with_detection)
            print("ping")
        
        if count == 250:

            img = cv_image.copy()

            print("Received an image! Please wait! Implementing Image Processing.")
            rospy.sleep(1)    
            img_with_detection,final_pose = feature_extract(img) 
            cv2.imwrite("/home/pinak/catkin_ws/src/img1.jpg",img_with_detection)
            errx = (final_pose[0][0]-init_pose[0][0]+final_pose[1][0]-init_pose[1][0]+final_pose[2][0]-init_pose[2][0]+final_pose[3][0]-init_pose[3][0])/4
            erry = (final_pose[0][1]-init_pose[0][1]+final_pose[1][1]-init_pose[1][1]+final_pose[2][1]-init_pose[2][1]+final_pose[3][1]-init_pose[3][1])/4
            error = math.sqrt(errx**2+erry**2)
        
        if count > 300 and error != 0:
            
            q1,q2 = j_ang
            q1 += 0.09
            q2 += 0.05
            print("Joint_pos",q1,q2)
            img = cv_image.copy()
            image_with_detection,final_pose = feature_extract(img)
            feat.append([final_pose[0][0],final_pose[0][1],final_pose[1][0],final_pose[1][1],final_pose[2][0],final_pose[2][1],final_pose[3][0],final_pose[3][1]])
            x_pos = (final_pose[0][0]+final_pose[1][0]+final_pose[2][0]+final_pose[3][0])/4
            y_pos = (final_pose[0][1]+final_pose[1][1]+final_pose[2][1]+final_pose[3][1])/4
            errx = (final_pose[0][0]-init_pose[0][0]+final_pose[1][0]-init_pose[1][0]+final_pose[2][0]-init_pose[2][0]+final_pose[3][0]-init_pose[3][0])/4
            erry = (final_pose[0][1]-init_pose[0][1]+final_pose[1][1]-init_pose[1][1]+final_pose[2][1]-init_pose[2][1]+final_pose[3][1]-init_pose[3][1])/4
            Le = np.array([[-1,0,0,0,0,y_pos],[0,-1,0,0,0,-1*x_pos]])
            Le_inv = np.linalg.pinv(Le)
            lamda = 0.05
            err = lamda*np.array([[errx],[erry]])
            Vc = np.matmul(Le_inv,err)
            print("Vc:",Vc)
            jv1,jv2 = inv_jacobian(q1,q2,Vc)
            error = math.sqrt(errx**2+erry**2)
            if (jv1>-0.003 and jv1<0.003) and (jv2>-0.003 and jv2<0.003):
                jv1 = 0.0
                jv2 = 0.0
                error = 0
                
            print("watch:",error)
            # print("see This", init_pose,final_pose)
            pub_q1_vel = rospy.Publisher('/vbmbot/joint1_velocity_controller/command', Float64, queue_size=10)
            pub_q2_vel = rospy.Publisher('/vbmbot/joint2_velocity_controller/command', Float64, queue_size=10)
            
            pub_q1_vel.publish(jv1)
			
            pub_q2_vel.publish(jv2)
            
        
        if error == 0:
            img = cv_image.copy()
            cv2.imshow("win",img)
            cv2.waitKey(0)
        with open('Visual_Servoing.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            # print("yup")
            for i in feat:
                writer.writerow(i)
                
        

        
            

    
        
def main(args):

    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    rospy.Subscriber('/vbmbot/joint_states', JointState, jointstate_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
   
if __name__ == '__main__':
    main(sys.argv)