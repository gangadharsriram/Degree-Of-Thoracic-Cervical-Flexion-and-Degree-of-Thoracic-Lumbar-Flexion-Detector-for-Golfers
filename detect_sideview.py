import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils


pose=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)


main_path=os.getcwd()
all_files=os.path.join(main_path,"side_view")
path=os.listdir(path=all_files)[0]
save_path=os.path.join(main_path,"output")
out=os.path.splitext(path)[0]+"_output"+os.path.splitext(path)[1]
output_file=os.path.join(save_path,out)

paths=os.path.join(all_files,path)

print(paths)
cap=cv2.VideoCapture(paths)
fps = int(cap.get(cv2.CAP_PROP_FPS))



width = int(cap.get(3))
height = int(cap.get(4))
video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


def cal_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    radians =np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle>360:
        angle=360-angle
    return angle

while True:
    _,frame=cap.read()
   
    if _ is False:
        break
    
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    
    result=pose.process(rgb)
    landmark=result.pose_landmarks.landmark    
    
    try:
        
        if (landmark[mp_pose.PoseLandmark.LEFT_EAR.value].visibility > landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility and 
        landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility and
        landmark[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility and 
        landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility and
        landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
        
        ):
            
            left_ear=landmark[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmark[mp_pose.PoseLandmark.LEFT_EAR.value].y
            left_sholder=landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            left_hip=landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
            left_knee=landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            left_elbow=landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
           
            
            cervical_1 =(left_sholder[0],left_ear[1])
            ang=tuple(np.multiply(cervical_1, [width, height]).astype(int))
            cv2.circle(rgb, ang, 5, (255,0,255), thickness=2)
            cv2.line(rgb, (tuple(np.multiply(left_ear, [width, height]).astype(int))), 
                        (tuple(np.multiply(cervical_1, [width, height]).astype(int))), 
                    (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(cervical_1, [width, height]).astype(int))),
                    (tuple(np.multiply(left_sholder, [width, height]).astype(int))), (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(left_sholder, [width, height]).astype(int))), 
                    (tuple(np.multiply(left_ear, [width, height]).astype(int))), (255, 255, 255), 2)
            
            
            
            lumbar_1 =(left_hip[0],left_sholder[1])
            ang_1=tuple(np.multiply(lumbar_1, [width, height]).astype(int))
            cv2.circle(rgb, ang_1, 5, (255,0,255), thickness=2)
            cv2.line(rgb, (tuple(np.multiply(left_sholder, [width, height]).astype(int))), 
                        (tuple(np.multiply(lumbar_1, [width, height]).astype(int))), 
                    (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(lumbar_1, [width, height]).astype(int))),
                    (tuple(np.multiply(left_hip, [width, height]).astype(int))), (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(left_sholder, [width, height]).astype(int))), 
                    (tuple(np.multiply(left_hip, [width, height]).astype(int))), (255, 255, 255), 2)
            
            
            
            x0,y0=tuple(np.multiply(left_ear, [width, height]).astype(int))
            
            cervical_angle_1=cal_angle(left_ear,left_sholder,cervical_1)
            x1,y1=tuple(np.multiply(left_sholder, [width, height]).astype(int))
            cv2.putText(rgb,str(int(cervical_angle_1)),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            lumbar_angle_1=cal_angle(left_sholder,left_hip,lumbar_1)
            x2,y2=tuple(np.multiply(left_hip, [width, height]).astype(int))
            cv2.putText(rgb,str(int(lumbar_angle_1)),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            
            elbow_angle_1=cal_angle(left_elbow,left_sholder,left_hip)
            
            
             
            cv2.line(rgb, (tuple(np.multiply(left_sholder, [width, height]).astype(int))), 
                    (tuple(np.multiply(left_elbow, [width, height]).astype(int))), (0, 255, 255), 2)
          
            
            cv2.putText(rgb,str(int(elbow_angle_1)),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            cv2.line(rgb, (x0,0), (x0,height), (255, 255,0), 3)
            cv2.line(rgb, (x1,0), (x1,height), (255, 255,0), 3)
            cv2.line(rgb, (x2,0), (x2,height), (255, 255,0), 3)
            
            
        
        #######################################################################################################################################################
        else:
            right_ear=landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].y
            right_sholder=landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            right_hip=landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee=landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            
            right_elbow=landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            
            
            cervical_1 =(right_sholder[0],right_ear[1])
            ang=tuple(np.multiply(cervical_1, [width, height]).astype(int))
            cv2.circle(rgb, ang, 5, (255,0,255), thickness=2)
            cv2.line(rgb, (tuple(np.multiply(right_ear, [width, height]).astype(int))), 
                        (tuple(np.multiply(cervical_1, [width, height]).astype(int))), 
                    (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(cervical_1, [width, height]).astype(int))),
                    (tuple(np.multiply(right_sholder, [width, height]).astype(int))), (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(right_sholder, [width, height]).astype(int))), 
                    (tuple(np.multiply(right_ear, [width, height]).astype(int))), (255, 255, 255), 2)
            
            
            lumbar_1 =(right_hip[0],right_sholder[1])
            ang_1=tuple(np.multiply(lumbar_1, [width, height]).astype(int))
            cv2.circle(rgb, ang_1, 5, (255,0,255), thickness=2)
            cv2.line(rgb, (tuple(np.multiply(right_sholder, [width, height]).astype(int))), 
                        (tuple(np.multiply(lumbar_1, [width, height]).astype(int))), 
                    (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(lumbar_1, [width, height]).astype(int))),
                    (tuple(np.multiply(right_hip, [width, height]).astype(int))), (255, 255, 255), 2)
            
            cv2.line(rgb, (tuple(np.multiply(right_sholder, [width, height]).astype(int))), 
                    (tuple(np.multiply(right_hip, [width, height]).astype(int))), (255, 255, 255), 2)
            
            
            x0,y0=tuple(np.multiply(right_ear, [width, height]).astype(int))
            
            right_angle_1=cal_angle(right_ear,right_sholder,cervical_1)
            x1,y1=tuple(np.multiply(right_sholder, [width, height]).astype(int))
            cv2.putText(rgb,str(int(right_angle_1)),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            right_angle_2=cal_angle(right_sholder,right_hip,lumbar_1)
            x2,y2=tuple(np.multiply(right_hip, [width, height]).astype(int))
            cv2.putText(rgb,str(int(right_angle_2)),(x2,y2),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
            
            
            elbow_angle_1=cal_angle(right_elbow,right_sholder,right_hip)
            
            
            cv2.line(rgb, (tuple(np.multiply(right_sholder, [width, height]).astype(int))), 
                    (tuple(np.multiply(right_elbow, [width, height]).astype(int))), (0, 255, 255), 2)
            
            
            cv2.putText(rgb,str(int(elbow_angle_1)),(x1,y1+30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            
            cv2.line(rgb, (x0,0), (x0,height), (255, 255,0), 3)
            cv2.line(rgb, (x1,0), (x1,height), (255, 255,0), 3)
            cv2.line(rgb, (x2,0), (x2,height), (255, 255,0), 3)
            
        
    except:
        pass
    
    
    
    rgb.flags.writeable=True
    rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    
    #mp_drawing.draw_landmarks(rgb,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(255,0,0,),thickness=3,circle_radius=3),
    #                          mp_drawing.DrawingSpec(color=(0,255,0,),thickness=3,circle_radius=3))
    
  
    
    
    
    video_writer.write(rgb)
    cv2.imshow("Output",rgb)
    
    if  cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    
