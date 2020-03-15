#from skimage.io import imread
#from skimage.util import crop
#from skimage.transform import rotate,resize,rescale
import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon
#from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data, Dataset,DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT
import math
import json
import csv


# scitsr数据集
class ScitsrDataset(Dataset):
    def __init__(self, root_path, transform=None, pre_transform=None):
        super(ScitsrDataset, self).__init__(root_path, transform, pre_transform)
        self.root_path = root_path
        self.jsonfile = os.path.join(self.root_path, "imglist.json")
        if os.path.exists(self.jsonfile):  # imglist.json去掉了一些有疑问的文件
            with open(self.jsonfile, "r") as read_file:
                self.imglist = json.load(read_file)
        else:  
            self.imglist = list(filter(lambda fn:fn.lower().endswith('.jpg') or fn.lower().endswith('.png') ,
                                       os.listdir(os.path.join(self.root_path,"img"))))
            self.imglist = self.check_all()
            with open(self.jsonfile, "w") as write_file:
                json.dump(self.imglist, write_file)
        self.graph_transform = GT.KNNGraph(k=6)
     
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
    
    def read_structure(self):
        return 
        
    def reset(self):
        pass
    
    def check_all(self):
        validlist=[]
        for idx in range(len(self.imglist)):
            print("*** file:",self.imglist[idx])
            structs, chunks = self.readlabel(idx)
            vi = self.check_chunks(structs, chunks)
            if vi==1:
                validlist.append(self.imglist[idx])
        print("valid:", len(validlist))
        return validlist
    
    # structs中不应该有空的cell，但是实际上可能有。空cell存在，会影响在chunk中的index。
    def remove_empty_cell(self,structs):
        structs.sort(key=lambda p: p["id"])
        news = []; idx = 0
        for st in structs:
            text = st["tex"].strip().replace(" ","")
            if text=="" or text=='$\\mathbf{}$':  #空的cell
                continue
            st["id"] = idx
            news.append(st)
            idx +=1
        return news

    def check_chunks(self,structs, chunks):
        structs = self.remove_empty_cell(structs)
        for st in structs:
            id = st["id"] 
            if id>=len(chunks):
                print("chunk index out of range.", id)
                return 0
            ch = chunks[id]
            txt1 = st["tex"].replace(" ","")
            txt2 = ch["text"].replace(" ","")
            #print(id, txt1," ", txt2)
            if txt1!=txt2:
                print(id, "mismatch:",txt1," ", txt2)
            if st["end_row"]-st["start_row"] !=0 or st["end_col"]-st["start_col"]!=0:
                print("span cells:", id)
        return 1
    
    def format_html(self,structs, chunks):
        rowcnt = max(structs, key=lambda p: p["end_row"])["end_row"]+1
        colcnt = max(structs, key=lambda p: p["end_col"])["end_col"]+1
        #print("row , col number:", rowcnt, colcnt)
        mat = [["<td></td>"]*colcnt for i in range(rowcnt)]
        for st in structs: # 填空
            mat[st["start_row"]][st["start_col"]] = "<td>" + st["tex"] + "</td>"
        html = ""
        #print(mat)
        for row in mat:
            html += "<tr>"+"".join(row)+"</tr>"
        return html    
        
    
    def readlabel(self,idx):
        imgfn = self.imglist[idx]
        structfn = os.path.join(self.root_path,"structure",os.path.splitext(os.path.basename(imgfn))[0] +".json")
        chunkfn = os.path.join(self.root_path,"chunk",os.path.splitext(os.path.basename(imgfn))[0]+".chunk")
        relfn = os.path.join(self.root_path,"rel",os.path.splitext(os.path.basename(imgfn))[0]+".rel")
        #print("***",chunkfn)
        if not os.path.exists(structfn) or  not os.path.exists(chunkfn):
            print("can't find files.")
            return
        with open(chunkfn, 'r') as f:
            chunks = json.load(f)['chunks']
        with open(structfn, 'r') as f:
            structs = json.load(f)['cells']
        #with open(relfn, 'r') as f:
        #    reader = csv.reader(f,delimiter='\t')
        #    rels = list(reader)
        #if (len(chunks) != len(structs)):
        #    print("chunks cells do not match. " + chunkfn)
        #    print(len(chunks),len(structs))
        
        #self.check_chunks(structs,chunks)
        return structs, chunks 
        
    
    def __len__(self):
        return len(self.imglist)
    
    
    def box_center(self, chkp):
        # x1, x2, y1, y2  in chkp
        return [(chkp[0]+chkp[1])/2, (chkp[2]+chkp[3])/2]
        
    def get_html(self, idx):
        structs, chunks = self.readlabel(idx)
        self.check_chunks(structs,chunks)
        html = self.format_html(structs, chunks)
        return html
    
    
    def cal_chk_limits(self, chunks):
        x_min = min(chunks, key=lambda p: p["pos"][0])["pos"][0]
        x_max = max(chunks, key=lambda p: p["pos"][1])["pos"][1]
        y_min = min(chunks, key=lambda p: p["pos"][2])["pos"][2]
        y_max = max(chunks, key=lambda p: p["pos"][3])["pos"][3]
        return [x_min,x_max,y_min,y_max,x_max-x_min,y_max-y_min]
    
    # 相对的位置。
    def pos_feature(self,chk,cl): 
        x1=10.0*(chk["pos"][0])/cl[4] 
        x2=10.0*(chk["pos"][1])/cl[4] 
        x3=10.0*(chk["pos"][2])/cl[5] 
        x4=10.0*(chk["pos"][3])/cl[5]
        x5 = (x1+x2)*0.5  # 中心点
        x6 = (x3+x4)*0.5
        x7 = x2-x1    # 文本宽度
        x8 = x4-x3    # 文本高度
        return [x1,x2,x3,x4,x5,x6,x7,x8]
        
    def get(self, idx):
        structs, chunks = self.readlabel(idx)
        cl = self.cal_chk_limits(chunks)
        #print(cl)
        #x = [chunks[st["id"]]["pos"] for st in structs]
        x,pos,tbpos=[],[],[]
        structs = self.remove_empty_cell(structs)
        for st in structs:
            id = st["id"] 
            chk = chunks[id]
            xt = self.pos_feature(chk,cl)
            x.append(xt)
            pos.append(xt[4:6])
            tbpos.append([st["start_row"],st["end_row"],st["start_col"],st["end_col"]])
        #y = []
        #for rel in rels:  # 先考虑同行的关系
        #    if rel[2][0] == '1':
        #        y.append([int(rel[0]),int(rel[1])])
                
            
        x = torch.FloatTensor(x)  
        pos = torch.FloatTensor(pos)  
        data = Data(x=x,pos=pos)
        data = self.graph_transform(data) # 构造图的连接
        y = self.cal_label(data, tbpos)
        data.y = torch.LongTensor(y)
        #print(data)
        return data
    
    def cal_label(self,data,tbpos): # 根据构造的图，计算边的标注。
        edges = data.edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
        y = []
        for i in range(edges.size()[1]):
#             y.append(self.if_same_col(edges[0,i], edges[1,i],tbpos))
            y.append(self.if_same_row(edges[0,i], edges[1,i],tbpos))
        return y
            
    def if_same_row(self,si,ti,tbpos):
        ss,se = tbpos[si][0], tbpos[si][1]
        ts,te = tbpos[ti][0], tbpos[ti][1]
        if (ss>=ts and se<=te):
            return 1
        if (ts>=ss and te<=se):
            return 1
        return 0
    
    def if_same_col(self,si,ti,tbpos):
        ss,se = tbpos[si][2], tbpos[si][3]
        ts,te = tbpos[ti][2], tbpos[ti][3]
        if (ss>=ts and se<=te):
            return 1
        if (ts>=ss and te<=se):
            return 1
        return 0
        
    
    def if_same_cell(self):
        pass

    
    
if __name__ == "__main__":  
    
    root_path = '/backup1/hz2/SciTSR/train'
    
    ds = ScitsrDataset(root_path)
    print(len(ds))
    #ds.check_all()
    #print(ds.get_html(76))
    test_loader = DataLoader(ds, batch_size=5 )
    for data in test_loader:
        print(data,data.num_graphs)
        #x = scatter_mean(data.x, data.batch, dim=0)
        #print(data.edge_index)
        #print(x.size())
        print("ratio:",data.y.sum().float()/data.y.size()[0])