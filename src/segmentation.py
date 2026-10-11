import cv2
import numpy as np

def _clean(mask, minimum):
    mask=(mask>0).astype(np.uint8)*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    out=np.zeros_like(mask)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=int(minimum): out[labels==i]=255
    return out

def _leaf(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    green=cv2.inRange(hsv,np.array([20,22,18]),np.array([105,255,255]))
    chromatic=cv2.inRange(hsv,np.array([5,18,25]),np.array([179,255,255]))
    labmask=cv2.inRange(lab[:,:,1],65,150)
    mask=_clean(cv2.bitwise_or(green,cv2.bitwise_and(chromatic,labmask)),max(30,int(rgb.shape[0]*rgb.shape[1]*.00015)))
    if cv2.countNonZero(mask)<rgb.shape[0]*rgb.shape[1]*.02:
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);_,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    n,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    if n>1:
        biggest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask=np.where(labels==biggest,255,0).astype(np.uint8)
    return mask

def analyze_leaf(rgb,sensitivity=1.0,color_space="HSV + CIELAB",min_component=120):
    rgb=np.asarray(rgb,dtype=np.uint8);hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    leaf=_leaf(rgb);f=float(sensitivity)
    sat=max(12,int(78-35*f));val=max(18,int(48-18*f))
    yellow=cv2.inRange(hsv,np.array([max(4,int(20-10*f)),sat,val]),np.array([int(38+8*f),255,255]))
    brown=cv2.inRange(hsv,np.array([0,sat,val]),np.array([int(20+8*f),255,int(180+40*f)]))
    lab_y=cv2.inRange(lab[:,:,2],int(137-9*f),255);lab_b=cv2.inRange(lab[:,:,1],int(135-8*f),255)
    hsvmask=cv2.bitwise_or(yellow,brown);labmask=cv2.bitwise_or(lab_y,lab_b)
    candidates=hsvmask if color_space=="HSV only" else labmask if color_space=="CIELAB only" else cv2.bitwise_or(hsvmask,labmask)
    mask=_clean(cv2.bitwise_and(candidates,leaf),min_component)
    lp=int(cv2.countNonZero(leaf));dp=int(cv2.countNonZero(cv2.bitwise_and(mask,leaf)))
    overlay=rgb.copy();region=mask>0;red=np.zeros_like(overlay);red[:]=(246,70,66)
    overlay[region]=(0.58*overlay[region]+0.42*red[region]).astype(np.uint8)
    contours,_=cv2.findContours(leaf,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(40,220,130),2)
    return {"leaf_mask":cv2.cvtColor(leaf,cv2.COLOR_GRAY2RGB),"disease_mask":cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB),"overlay":overlay,"hsv_view":cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),"lab_view":cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),"leaf_pixels":lp,"disease_pixels":dp,"affected_pct":100*dp/lp if lp else 0.0}

def make_pipeline_image(rgb,result):
    panels=[(rgb,"Input image"),(result["leaf_mask"],"Leaf mask"),(result["disease_mask"],"Candidate mask"),(result["overlay"],"Overlay")]
    height=220;thumbs=[]
    for image,label in panels:
        h,w=image.shape[:2];nw=max(1,int(w*height/max(1,h)))
        thumb=cv2.resize(image,(nw,height),interpolation=cv2.INTER_AREA)
        canvas=np.full((height+38,nw,3),248,np.uint8);canvas[:height]=thumb
        cv2.putText(canvas,label,(7,height+25),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,65,48),1,cv2.LINE_AA);thumbs.append(canvas)
    gap=np.full((height+38,12,3),248,np.uint8);out=thumbs[0]
    for t in thumbs[1:]:out=np.concatenate([out,gap,t],axis=1)
    return out
