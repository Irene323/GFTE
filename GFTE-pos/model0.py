import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64*2, 64)
        self.lin2 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x=x.cuda()
        edge_index = edge_index.cuda()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # combine node features
        x1=x[edge_index[0]]
        x2=x[edge_index[1]]
        xpair =torch.cat((x1,x2),dim=1)
        xpair = F.relu(self.lin1(xpair))
        xpair = (self.lin2(xpair))
        return F.log_softmax(xpair, dim=1)

if __name__ == "__main__":  
    print("test")
    