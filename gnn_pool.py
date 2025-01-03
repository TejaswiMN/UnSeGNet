import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
# self.model = GNNpool(self.feats_dim=384, conv_hidden=64, mlp_hidden=32, num_clusters=2, self.device, activation, loss_type, conv_type).to(self.device)

class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, device, activ="silu", loss_type="DMON", conv_type="ARMA"):
        """
        implementation of mincutpool model from: https://arxiv.org/pdf/1907.00481v6.pdf
        @param input_dim: Size of input nodes features
        @param conv_hidden: Size Of conv hidden layers
        @param mlp_hidden: Size of mlp hidden layers
        @param num_clusters: Number of cluster to output
        @param device: Device to run the model on
        @param activ: Activation function to use. Enum: ["deepcut_activation", "relu", "silu", "gelu", "selu"]
        @param loss_type: Loss function to use. Enum: ["DMON", "NCUT"]
        """
        super(GNNpool, self).__init__()
        self.device = device
        self.activ = activ
        self.num_clusters = 2 #num_clusters
        self.mlp_hidden = mlp_hidden
        self.convtype = conv_type
        if loss_type not in ["DMON", "NCUT"]:
            raise ValueError(f'Loss type: {loss_type} is not supported')
        self.loss_type = loss_type

        if activ == "deepcut_activation":
            act = 'relu'
            nn_activ = nn.ReLU()
            self.f_act = F.elu
        elif activ == "relu":
            act = 'relu'
            nn_activ = nn.ReLU()
            self.f_act = F.relu
        elif activ == "silu":
            act = 'silu'
            nn_activ = nn.SiLU()
            self.f_act = F.silu
        elif activ == "gelu":
            act = 'gelu'
            nn_activ = nn.GELU()
            self.f_act = F.gelu
        elif activ == "selu":
            act = 'selu'
            nn_activ = nn.SELU()
            self.f_act = F.selu
        else:
            raise ValueError("Activation function not supported")
            

        # GNN conv
        if conv_type == "ARMA":
            self.convs = pyg_nn.ARMAConv(input_dim, conv_hidden, num_stacks=2, num_layers=4, act=nn_activ,\
                dropout=0.4,shared_weights=False)
        elif conv_type == "GCN":
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act=act)
        elif conv_type == "GAT":
            # self.convs1 = pyg_nn.GCN(input_dim, 128, 1, act=act)
            # self.convs2 = pyg_nn.GATConv(128, 64, heads=2, concat=False, dropout=0.4, negative_slope=0.2)
            # self.convs3 = pyg_nn.GCN(64, conv_hidden, 1, act=act)
            self.convs = pyg_nn.GATConv(input_dim, conv_hidden, heads=2, concat=False, dropout=0.4, negative_slope=0.2)
        elif conv_type == "GCNGAT":
            self.convs1 = pyg_nn.GCN(input_dim, 128, 1, act=act)
            self.convs2 = pyg_nn.GATConv(128, 64, heads=3, concat=False, dropout=0.4, negative_slope=0.2)
        elif conv_type == "EnsembleGAT":
            self.num_layers = 3
            self.gat_layers = nn.ModuleList([   
                pyg_nn.GATConv(input_dim, conv_hidden, heads=2, concat=False, dropout=0.4, negative_slope=0.2) for _ in range(self.num_layers)
            ])
            self.final_linear = nn.Linear(conv_hidden * self.num_layers, conv_hidden)

        else:
            raise ValueError("Conv type not supported")

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn_activ, nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))

    def forward(self, data, A):
        """
        forward pass of the model
        @param data: Graph in Pytorch geometric data format
        @param A: Adjacency matrix of the graph
        @return: Adjacency matrix of the graph and pooled graph (argmax of S)
        """
        x, edge_index, edge_atrr = data.x, data.edge_index, data.edge_attr
        if self.convtype == "GAT":
            # x = self.convs1(x, edge_index, edge_atrr)
            # x = self.convs2(x, edge_index, edge_atrr)
            # x = self.convs3(x, edge_index, edge_atrr)
            x = self.convs(x, edge_index, edge_atrr)
        elif self.convtype == "GCNGAT":
            x = self.convs1(x, edge_index, edge_atrr)
            x = self.convs2(x, edge_index, edge_atrr)
        elif self.convtype == "EnsembleGAT":
            outputs = []
            for layer in self.gat_layers:
                outputs.append(layer(x, edge_index))
            x = torch.cat(outputs, dim=1)
            #print(f"Shape after concatenation: {x.shape}")  # Check shape after concatenation
            x = self.final_linear(x)
            #print(f"Shape after final linear: {x.shape}")
        else:
            x = self.convs(x, edge_index, edge_atrr)  # applying con5v
        x = self.f_act(x)

        # pass feats through mlp
        H = self.mlp(x)
        # cluster assignment for matrix S
        S = F.softmax(H)

        return A, S

    def loss(self, A, S):
        """
        loss calculation, relaxed form of Normalized-cut
        @param A: Adjacency matrix of the graph
        @param S: Polled graph (argmax of S)
        @return: loss value
        """
        if self.loss_type == "NCUT":
            # cut loss
            A_pool = torch.matmul(torch.matmul(A, S).t(), S)
            num = torch.trace(A_pool)

            D = torch.diag(torch.sum(A, dim=-1))
            D_pooled = torch.matmul(torch.matmul(D, S).t(), S)
            den = torch.trace(D_pooled)
            mincut_loss = -(num / den)
            # orthogonality loss
            St_S = torch.matmul(S.t(), S)
            I_S = torch.eye(self.num_clusters, device=self.device)
            ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S / torch.norm(I_S))

            return mincut_loss + ortho_loss
        elif self.loss_type == "DMON":
            C = S
            d = torch.sum(A, dim=1)
            m = torch.sum(A)
            B = A - torch.ger(d, d) / (2 * m)
            
            I_S = torch.eye(self.num_clusters, device=self.device)
            k = torch.norm(I_S)
            n = S.shape[0]
            
            modularity_term = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))
            
            collapse_reg_term = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C.t(), dim=0), p='fro')) - 1
            
            return modularity_term + collapse_reg_term
