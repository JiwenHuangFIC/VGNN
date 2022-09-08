import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TensorFusionLayer(nn.Module):
    """Tensor-based Fusion Method"""
    def __init__(self, in_features, out_features, dropout):
        super(TensorFusionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(2 * in_features, out_features)
        self.T_tensor = nn.Parameter(torch.zeros(size=(in_features, in_features, in_features)))
        nn.init.xavier_uniform_(self.T_tensor.data)

    def forward(self, inputdata):
        inputdata = F.dropout(inputdata, self.dropout, training=self.training)
        inputdata_l = inputdata.unsqueeze(2)
        inputdata_r = inputdata.unsqueeze(1)
        x_inter = torch.matmul(inputdata_l, inputdata_r)
        R = torch.tanh(torch.matmul(inputdata, self.T_tensor.transpose(0, 1).reshape(inputdata.size(1), -1)).reshape(inputdata.size(0), inputdata.size(1), inputdata.size(1)))
        x_inter_R = torch.mul(x_inter, R).sum(2)
        # concat
        x_concat = torch.cat([inputdata, x_inter_R], dim=1)
        x_return = F.elu(self.fc(x_concat))
        return x_return, R


class ImplicitLayer(nn.Module):
    """Attention Mechanism for Implicit Relation"""
    def __init__(self, in_features, out_features, dropout, alpha):
        super(ImplicitLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(2 * in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.size()[0]  # number of nodes
        a_input = self._prepare_attentional_mechanism_input(h)
        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        mask_adj = torch.ones(N, N, device=h.device) - torch.eye(N, N, device=h.device)
        attention = torch.mul(mask_adj, attention)
        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _prepare_attentional_mechanism_input(self, h):
        input_l = torch.matmul(h, self.W[:self.in_features, :])
        input_r = torch.matmul(h, self.W[self.in_features:, :])
        return input_l.unsqueeze(1) + input_r


class Self_attention(nn.Module):
    """
    To incorporate the global influence
    """
    def __init__(self, in_features, out_features, dropout):
        super(Self_attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.Q_fc = nn.Linear(in_features, out_features)
        self.K_fc = nn.Linear(in_features, out_features)
        self.V_fc = nn.Linear(in_features, out_features)

    def forward(self, inputdata):

        Q = self.Q_fc(inputdata)
        K = self.K_fc(inputdata)
        V = self.V_fc(inputdata)
        Scores = torch.matmul(Q, K.T) / math.sqrt(Q.size(-1))
        Scores_softmax = F.softmax(Scores, dim=-1)
        Scores_softmax = F.dropout(Scores_softmax, self.dropout, training=self.training)
        Market_Signals = torch.matmul(Scores_softmax, V)
        return Market_Signals

class Relation_Attention(nn.Module):
    """
    calculating the importance of each pre-defined relation.
    """
    def __init__(self, in_features, out_features, dropout):
        super(Relation_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)


    def forward(self, Adj, inputdata):

        attention_temp = torch.matmul(inputdata, self.W)
        attention = torch.sigmoid(torch.matmul(attention_temp, attention_temp.T))
        attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.mul(Adj, attention)

class ExplicitLayer(nn.Module):
    """
    To combine the varying contributions of multi-relations to form the explicit link.
    """
    def __init__(self, in_features, out_features, dropout):
        super(ExplicitLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.self_attention = Self_attention(in_features, out_features, dropout)
        self.relation_attention_ind = Relation_Attention(out_features, out_features, dropout)
        self.relation_attention_loc = Relation_Attention(out_features, out_features, dropout)

    def forward(self, A_Ind, A_Loc, inputdata):

        # Obtain the global market information
        market_Signals = self.self_attention(inputdata)  # shape:[number of firms, the dimension of the global market information]
        market_Signals = F.dropout(market_Signals, self.dropout, training=self.training)
        # Obtain the adjacency matrix of industry Relation under the Attention mechanism
        A_ind = self.relation_attention_ind(A_Ind, market_Signals)
        # Get the adjacency matrix of location Relation under the Attention mechanism
        A_loc = self.relation_attention_loc(A_Loc, market_Signals)
        return A_ind + A_loc

class AttributeGate(nn.Module):

    """Gate Mechanism for attribute passing"""

    def __init__(self, in_features, out_features, dropout):
        super(AttributeGate, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(2 * in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.b = nn.Parameter(torch.empty(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data)

    def forward(self, h):
        attribute_gate = self._prepare_attentional_mechanism_input(h)
        attribute_gate = torch.tanh(attribute_gate.add(self.b))
        attribute_gate = F.dropout(attribute_gate, self.dropout, training=self.training)
        return attribute_gate

    def _prepare_attentional_mechanism_input(self, h):
        input_l = torch.matmul(h, self.W[:self.in_features, :])
        input_r = torch.matmul(h, self.W[self.in_features:, :])
        return input_l.unsqueeze(1) + input_r