import numpy as np
import time


def get_rec(img):
  
  h,w = img.shape[0],img.shape[1]
  sum = img.sum()
  weight = img/(sum + 1e-7)
  val = img[img>0]
  if len(val) == 0:
    c = 0
  else:
    c = np.median(val)
    c = np.average(val[val>=c])
  cx = weight * (np.arange(h).reshape(h,1))
  cx = cx.sum()
  cy = weight * (np.arange(w).reshape(1,w))
  cy = cy.sum()
  cx = np.round(2*cx)/2
  cy = np.round(2*cy)/2

  weight_x1,weight_y1 = img,img
  weight_x2 = 2*img[0:-1]*img[1:]/(img[0:-1] + img[1:] + 1e-7) 
  weight_y2 = 2*img[:,0:-1]*img[:,1:]/(img[:,0:-1] + img[:,1:] + 1e-7)
  if int(cx) == cx:
    additionx = img[int(cx)].sum()
  else:
    additionx = weight_x2[int(np.floor(cx))].sum()
  if int(cy) == cy:
    additiony = img[:,int(cy)].sum()
  else:
    additiony = weight_y2[:,int(np.floor(cy))].sum()
  sumx = weight_x1.sum() + weight_x2.sum() + additionx + 1e-7
  sumy = weight_y1.sum() + weight_y2.sum() + additiony + 1e-7
  weight_x1 = weight_x1/sumx
  weight_x2 = weight_x2/sumx
  weight_y1 = weight_y1/sumy
  weight_y2 = weight_y2/sumy 
  dx = (weight_x1 * abs(np.arange(h)-cx).reshape(h,1)).sum() + (weight_x2 * abs(np.arange(0.5,h-1,1)-cx).reshape(h-1,1)).sum()
  dy = (weight_y1 * abs(np.arange(w)-cy).reshape(1,w)).sum() + (weight_y2 * abs(np.arange(0.5,w-1,1)-cy).reshape(1,w-1)).sum()
   
  
  return cx,cy,4*dx,4*dy,c

def get_cor(cx,cy,dx,dy,mx,my):
  
  x1 = cx - dx/2 
  x2 = cx + dx/2 
  y1 = cy - dy/2 
  y2 = cy + dy/2 
  x1 = np.clip(x1,0,mx)
  x2 = np.clip(x2,0,mx)
  y1 = np.clip(y1,0,my)
  y2 = np.clip(y2,0,my)
  return int(np.round(x1)),int(np.round(y1)),int(np.round(x2)),int(np.round(y2))

#get the loss and the ideal rectangle as well as the four parameters of the ideal rectangle

def get_metric(img):
  shape = img.shape
  fimg = img/(img.max() + 1e-7)
  res = get_rec(fimg)
  cx,cy,dx,dy,c = res
  x1,y1,x2,y2 = get_cor(cx,cy,dx,dy,img.shape[0]-1,img.shape[1]-1)
  rec = np.zeros(img.shape)
  rec[x1:x2+1,y1:y2+1] = c

  gapx = shape[0]/32
  gapy = shape[1]/32
  rdx = img.sum(1)>1e-1
  rdy = img.sum(0)>1e-1
  rdx = max(rdx.sum()-gapx,0)
  rdy = max(rdy.sum()-gapy,0)
  dis = abs(fimg - rec).sum()

  loss = (dis+1e-2) / (np.log(1+1e-2+rdx) * np.log(1+1e-2+rdy))

  return loss,rec,cx,cy,dx,dy

