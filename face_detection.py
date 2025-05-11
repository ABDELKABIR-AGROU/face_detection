import mediapipe as mp
import time



mp_face_detection=mp.solutions.face_detection

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('gad lfrin')

with mp_face_detection.FaceDetection(min_detection_confidence=0.25) as face_detection:
    
    while True:
        ti=time.time()
        res,frame=cap.read()
        #frame=cv2.cv2.medianBlur(frame,3)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results=face_detection.process(frame)
        
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        if results.detections:
            
           for id,detection in enumerate(results.detections):
               
               box=detection.location_data.relative_bounding_box
               
               h,w=frame.shape[:2]
               
               bbox=int(box.xmin*w),int(box.ymin*h),int(box.width*w),int(box.height*h)
               
               cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0),3)
              
              
        cv2.putText(frame, str(round(time.time()-ti,3)), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
           
        cv2.imshow("tag",frame)
        c=cv2.waitKey(1)
        if c==ord('x'):
            break
cap.release()
cv2.destroyAllWindows()
