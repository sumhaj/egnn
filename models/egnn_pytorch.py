import torch
from torch import nn

class EGCL_Sparse(nn.module):
    def __init__(self, input_nf_dim, output_nf_dim, hidden_nf_dim, ef_dim=0, residual=True, tanh=False, coord_agg='mean'):
        super().__init__()
        self.residual = residual
        self.coord_agg = coord_agg
        self.tanh = tanh

        self.edge_mlp = nn.Sequential(
                            nn.Linear(2*input_nf_dim + 1 + ef_dim, hidden_nf_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_nf_dim, hidden_nf_dim),
                            nn.SiLU(),
                        )
        
        coord_mlp = [
                        nn.Linear(hidden_nf_dim, hidden_nf_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_nf_dim, 1),
                    ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        
        self.node_mlp = nn.Sequential(
                            nn.Linear(input_nf_dim + hidden_nf_dim, hidden_nf_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_nf_dim, output_nf_dim),
                        )
        
    def edge_model(self, source, target, coord_diff_norm, edge_attr):
        
        if edge_attr is None:
            edge_feature = self.edge_mlp(torch.cat([source, target, coord_diff_norm], dim=1))
        else:
            edge_feature = self.edge_mlp(torch.cat([source, target, coord_diff_norm, edge_attr], dim=1))

        return edge_feature
    
    def coord_model(self, edge_feature, coord, coord_diff, edge_index):
        row, _ = edge_index

        weighted_coord_diff = coord_diff * self.coord_mlp(edge_feature)

        if self.coord_agg == 'mean':
            coord = coord + torch.zeros_like(coord).scatter_reduce(0, row, weighted_coord_diff, reduce='mean', include_self=False)
        elif self.coord_agg == 'sum':
             coord.scatter_reduce(0, row, weighted_coord_diff, reduce='sum', include_self=True)
        else:   
            return Exception('Unknown coords_agg parameter, Only \'mean\' and \'sum\' are valid parameters' % self.coords_agg)
        
        return coord

        
    def node_model(self, edge_feature, h, edge_index):
        row, _ = edge_index
        agg = torch.zeros_like(h).scatter_reduce(0, row, edge_feature, reduce='sum')
        out = self.node_mlp(torch.cat([h, agg], dim=1))
        if self.residual:
            out = h + out
        return out

    def forward(self, coord, h, edge_index, edge_attr=None):
        row, column = edge_index
        coord_diff = coord[row] - coord[column]
        coord_diff_norm = torch.sum(coord_diff ** 2, dim=1)
       
        edge_feature = self.edge_model(h[row], h[column], coord_diff_norm, edge_attr)
        coord = self.coord_model(edge_feature, coord, coord_diff, edge_index)
        h = self.node_model(edge_feature, h, edge_index)

        return coord, h
        

