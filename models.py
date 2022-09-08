import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import TensorFusionLayer, ImplicitLayer, ExplicitLayer, AttributeGate

class VGNN(nn.Module):
    def __init__(self, nfeat, nhid, hidden_RNN, hidden_spillover, nclass, dropout, alpha, TensorFusion=True, Implicit=True, Explicit=True):
        """Tensor-based Feature Fusion Module"""
        super(VGNN, self).__init__()
        self.tensorfusion_c = TensorFusion
        self.implicit_c = Implicit
        self.explicit_c = Explicit
        self.dropout = dropout
        self.nhid = nhid
        self.nclass = nclass
        self.alpha = alpha

        self.fusion = TensorFusionLayer(nfeat, nhid, dropout)
        self.fc_no_fusion = nn.Linear(nfeat, nhid)
        self.rnn = nn.GRU(nhid, hidden_RNN)
        self.implicit = ImplicitLayer(hidden_RNN, nhid, dropout, alpha)
        self.explicit = ExplicitLayer(hidden_RNN, nhid, dropout)
        self.attribute = AttributeGate(hidden_RNN, hidden_spillover, dropout)
        self.out_1 = nn.Linear(hidden_RNN + hidden_spillover, 16)
        # self.out_2 = nn.Linear(32, 16)
        self.out_3 = nn.Linear(16, nclass)

        self.graph_W = nn.Parameter(torch.empty(size=(hidden_RNN, hidden_spillover)))
        nn.init.xavier_uniform_(self.graph_W.data)

    def forward(self, A_Ind, A_Loc, inputdata):

        # Obtain the fusion features
        if self.tensorfusion_c:
            n_window = inputdata.size(0)
            n_firm = inputdata.size(1)
            regularization_R = 0
            x = torch.cat([self.fusion(day_data)[0] for day_data in inputdata], dim=0)
            for day_data in inputdata:
                regularization_R += torch.sum(torch.abs(self.fusion(day_data)[1]))
            x = x.reshape(n_window, n_firm, -1)
        else:
            x = self.fc_no_fusion(inputdata)
        # lstm
        x = F.dropout(x, self.dropout, training=self.training)
        out, h = self.rnn(x)
        # Sequential Embeddings
        x = h[0]
        x = F.dropout(x, self.dropout, training=self.training)
        # implicit link
        if self.implicit_c:
            implicit_relation = self.implicit(x)
        else:
            implicit_relation = torch.zeros(x.size(0), x.size(0), device=x.device)
        # explicit link
        if self.explicit_c:
            explicit_relation = self.explicit(A_Ind, A_Loc, x)
        else:
            explicit_relation = torch.zeros(x.size(0), x.size(0), device=x.device)
        # final link
        relation = implicit_relation + explicit_relation
        # Softmax
        #############
        zero_mat = -9e10 * torch.ones_like(relation, device=inputdata.device)
        A = torch.where(relation > 0, relation, zero_mat)
        A = F.softmax(A, dim=1)
        ###############

        # Obtain the Spillovers Embeddings
        Mh = torch.matmul(x, self.graph_W)
        # Obtain the attribute gate
        attribute_gate = self.attribute(x)
        gate_Mh = torch.mul(Mh, attribute_gate)
        # aggregation
        H = torch.matmul(A.view(x.size(0), 1, x.size(0)), gate_Mh).view(x.size(0), -1)
        # concat: Spillovers Embeddings || Sequential Embeddings (|| denotes concatenation)
        H = torch.cat([H, x], -1)
        # output mapping
        H = F.elu(self.out_1(H))
        H = F.dropout(H, self.dropout, training=self.training)
        # H = F.elu(self.out_2(H))
        # H = F.dropout(H, self.dropout, training=self.training)
        H = self.out_3(H)
        return H, implicit_relation, regularization_R
