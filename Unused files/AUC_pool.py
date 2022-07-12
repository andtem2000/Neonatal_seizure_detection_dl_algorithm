import numpy as np
from Neonatal_Seizure_Resnext_algorithm.enframe import enframe
from Neonatal_Seizure_Resnext_algorithm.polyarea import simpoly


#D_val = sio.loadmat("D_val.mat")
#D_val = D_val["D_val"][0]
#epoch_map = sio.loadmat("epoch_map.mat")
#epoch_map = epoch_map["epoch_map"][0]
#D_val1 = sio.loadmat("D_val1.mat")
#D_val1 = D_val1["D_val"][0]
#epoch_map1 = sio.loadmat("epoch_map1.mat")
#epoch_map1 = epoch_map1["epoch_map"][0]

def calc_roc(epoch_map, D_val):
    D_val = np.asarray(D_val)
    collar = 30
    ctr = 0
    sens = []
    spec = []
    prec = []
    prec2 = []
    
    float_values = [float(x)/10000 for x in range(0,10025,25)]    

    #pdb.set_trace()
 
    for value in float_values:
        decision = np.zeros((1,len(D_val)))
        for idx in np.where(D_val>=1-value)[0]:
            decision[0,idx] = int(1)
        
        shiftmean = (9-1)/2
        my_dec = np.zeros((len(decision[0]) + shiftmean*2))
        pt1 = decision[:,:shiftmean] 
        my_dec[:len(pt1[0])] = np.fliplr([pt1])[0]
        my_dec[len(pt1[0]):len(pt1[0])+ len(decision[0])] = decision[0]
        pt2 = decision[:,-shiftmean:]
        my_dec[len(pt1[0])+ len(decision[0]):] = np.fliplr([pt2])[0]
        
        
        decision = np.zeros((1,len(D_val)))
        for zxc in range(len(decision)):
            aaa = enframe(np.array(my_dec), 9, 1)
            for idx in np.where(np.sum(aaa,1) == 9)[0]:
                # for idx in [i for i,v in enumerate([val for val in np.sum(aaa,1)]) if v == 9]:
                decision[0,idx] = 1
        #pdb.set_trace() 
        tmp = np.where(np.array(decision[0]) == 1)
        finidx = []
        
        for kkk in range(collar+1):
            a =tmp[0]-kkk# [val-kkk for val in tmp[0]]
            b =tmp[0]+kkk# [val +kkk for val in tmp[0]]
            finidx.extend(a)          
            finidx.extend(b)
            finidx = list(set(finidx))
        negative_values = np.where(np.array(finidx) <0)
        for neg in negative_values[0]:
            #print(neg)
            finidx[neg] = 0
        positive_values = np.where(np.array(finidx) >= len(decision[0]))
        for pos in positive_values[0]:
            finidx[pos] = len(decision[0])-1
        for pred_idx in finidx:
            decision[0][pred_idx] = 1
        #pdb.set_trace() 
       # print(finidx)
        
        TN = 0
        FN = 0
        FP = 0
        TP = 0
        #calc_vec1 = epoch_map[:len(epoch_map)/2] + decision[:,:len(epoch_map)/2]*2
        calc_vec = epoch_map + decision*2
        TN = TN + float(len(np.where(np.array(calc_vec[0]) == 0)[0]))
        FN = FN + float(len(np.where(np.array(calc_vec[0]) == 1)[0]))
        FP = FP + float(len(np.where(np.array(calc_vec[0]) == 2)[0]))
        TP = TP + float(len(np.where(np.array(calc_vec[0]) == 3)[0]))
        del calc_vec
        
        try:
            sens.append(float(TP/(TP + FN)))
        except ZeroDivisionError:
            sens.append(np.nan)
        try:            
            spec.append(float(TN/(TN + FP)))
        except ZeroDivisionError:
            spec.append(np.nan)
        try:
            prec.append(float(TP/(TP + FP)))
        except ZeroDivisionError:
            prec.append(np.nan)
        try:
            prec2.append(float(TN/(TN + FN)))
        except ZeroDivisionError:
            prec2.append(np.nan)
        del TN, TP, FN, FP
        ctr = ctr +1
        
       # print datetime.now() - startTime2
    #print(prec)
   # aind = [i for i,v in enumerate([val for val in np.isfinite(prec)]) if v != 0] 
   # bind = [i for i,v in enumerate([val for val in np.isfinite(sens)]) if v != 0] 
   # indcom = [val for val in aind if val in bind]
    x = [0, 1]
    x.extend(spec)
    x.extend([0])
    y = [0,0]
    y.extend(sens)
    y.extend([1])
    roc_area = simpoly(x, y)
   # x = [0,0]
   # x.extend(sens[indcom])
   # x.extend([1])
   # y = [0, 1]
   # y.extend(prec[indcom]
   # y.extend([0])
   # roc_pr = simploy(x,y)
    #print("Test ROC area = %f " % (roc_area))
    return(roc_area)    
        
