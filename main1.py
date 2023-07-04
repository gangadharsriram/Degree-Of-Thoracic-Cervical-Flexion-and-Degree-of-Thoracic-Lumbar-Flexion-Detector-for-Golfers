import mediapipe as mp
import cv2
import numpy as np  
import os



main_path=os.getcwd()
all_files=os.path.join(main_path,"front_view")
path=os.listdir(path=all_files)[0]
save_path=os.path.join(main_path,"output")
out=os.path.splitext(path)[0]+"_output"+os.path.splitext(path)[1]
output_file=os.path.join(save_path,out)

paths=os.path.join(all_files,path)

cap=cv2.VideoCapture(paths)


width = int(cap.get(3))
height = int(cap.get(4))

fps = int(cap.get(cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils


op=mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
while True:
    _,frame=cap.read()
    
    if _ is False:
        break
    
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    
    result=op.process(rgb)
    
    out=result.pose_landmarks.landmark
    
    nose=out[mp_pose.PoseLandmark.NOSE.value].x,out[mp_pose.PoseLandmark.NOSE.value].y
    r_sho=out[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,out[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    l_sho=out[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,out[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    
    l_hip=out[mp_pose.PoseLandmark.LEFT_HIP.value].x,out[mp_pose.PoseLandmark.LEFT_HIP.value].y
    r_hip=out[mp_pose.PoseLandmark.RIGHT_HIP.value].x,out[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    
    
    nose=tuple(np.multiply(nose, [width, height]).astype(int))
    
    r_sho=tuple(np.multiply(r_sho, [width, height]).astype(int))
    l_sho=tuple(np.multiply(l_sho, [width, height]).astype(int))        
    
    r_hip=tuple(np.multiply(r_hip, [width, height]).astype(int))
    l_hip=tuple(np.multiply(l_hip, [width, height]).astype(int))
    
    
        
    chest=[]
    if r_sho[0]<l_sho[0]:
        l=int(((l_sho[0]-r_sho[0])/2)+r_sho[0])
        w=int((l_sho[1]-r_sho[1])+r_sho[1])    
        ty=((l,w))
        chest.append(ty)
    else:
        l=int(((r_sho[0]-l_sho[0])/2)+l_sho[0])
        w=int((r_sho[1]-l_sho[1])+l_sho[1])    
        ty=((l,w))
        chest.append(ty)
        
    stomach=[]
    if r_hip[0]<l_hip[0]:
        l=int(((l_hip[0]-r_hip[0])/2)+r_hip[0])
        w=int((l_hip[1]-r_hip[1])+r_hip[1])    
        ty=((l,w))
        stomach.append(ty)
    else:
        l=int(((r_hip[0]-l_hip[0])/2)+l_hip[0])
        w=int((r_hip[1]-l_sho[1])+l_hip[1])    
        ty=((l,w))
        stomach.append(ty)
    
    cv2.circle(rgb, nose, 15, (0,255,0), thickness=3)
    
    
    cv2.circle(rgb, r_sho, 15, (255,0,0), thickness=3)
    cv2.circle(rgb, l_sho, 10, (255,0,0), thickness=3)
    
    cv2.circle(rgb, r_hip, 15, (0,0,255), thickness=3)
    cv2.circle(rgb, l_hip, 15, (0,0,255), thickness=3)
    
    cv2.circle(rgb, chest[-1], 10, (255,255,0), thickness=8)
    cv2.circle(rgb, stomach[-1], 10, (0,255,255), thickness=8)
    
    cv2.line(rgb, nose, chest[-1], (255, 255, 255), 2)        
    cv2.line(rgb,chest[-1],stomach[-1], (255, 255, 255), 2)        
    
             
    cv2.line(rgb, r_sho, chest[-1], (255, 255, 255), 2)        
    cv2.line(rgb, chest[-1],l_sho, (255, 255, 255), 2)
    
    cv2.line(rgb, r_hip, stomach[-1], (255, 255, 255), 2)        
    cv2.line(rgb, stomach[-1],l_hip, (255, 255, 255), 2)
    
    
    
    rgb.flags.writeable = True
    rgb=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    
    """
    mp_drawing.draw_landmarks(rgb,result.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,0,0,),thickness=3,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,255,0,),thickness=3,circle_radius=3))
    """
    video_writer.write(rgb)
    pp=cv2.resize(rgb, (480,640))
    cv2.imshow("Output", rgb)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
