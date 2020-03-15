import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class TbNet(torch.nn.Module):
    def __init__(self, num_node_features, vocab_size, num_text_features, num_classes):
        super(TbNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.embeds = nn.Embedding(vocab_size, num_text_features)
        self.rnn = nn.GRU(num_text_features, 64, bidirectional=False, batch_first=True)
        #self.text_simple_lin = torch.nn.Linear(num_text_features*10, num_text_features) # 临时措施。应该用lstm
        self.lin1 = torch.nn.Linear(64*2, 64)
        self.lin_text = torch.nn.Linear(64*2, 64)
        self.lin_final = torch.nn.Linear(64*2, num_classes)

    def forward(self, data):
        x, edge_index, xtext = data.x, data.edge_index, data.xtext
        x=x.cuda()
        edge_index = edge_index.cuda()
        xtext = xtext.cuda()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #print(xtext.size(),xtext.dtype)
        xtext = self.embeds(xtext)
        textout, _ = self.rnn(xtext)
        textout = textout[:,-1, :]        
        #print(textout.size())
        #xtext = xtext.reshape(xtext.size()[0],-1)
        #xtext = self.text_simple_lin(xtext)
        #x=x+xtext  # 融合
        # combine node features
        x1=x[edge_index[0]]
        x2=x[edge_index[1]]
        xpair =torch.cat((x1,x2),dim=1)
        xpair = F.relu(self.lin1(xpair))
        # combine text node feature
        x1text = textout[edge_index[0]]
        x2text=textout[edge_index[1]]
        xpairtext =torch.cat((x1text,x2text),dim=1)
        xpairtext = F.relu(self.lin_text(xpairtext))
        
        xfin = torch.cat((xpair,xpairtext),dim=1)
        xfin = (self.lin_final(xfin))
        return F.log_softmax(xfin, dim=1)

if __name__ == "__main__":  
    print("test")
    