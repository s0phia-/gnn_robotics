# %%
import torch

t1 = torch.tensor([[0,1],[1,2]])
t2 = torch.tensor([[0,1],[1,2],[0,2]])

edge_attr = torch.zeros_like(t2)
edge_attr[:, 1] = 1
edge_attr
# %%
for edge in t1:
    # Convert tensors to tuples for comparison
    edge_tuple = tuple(edge.tolist())
    for i, t2_edge in enumerate(t2):
        if tuple(t2_edge.tolist()) == edge_tuple:
            edge_attr[i,0] = 1
            break

edge_attr
# %%
t3 = torch.tensor([[0,1,2,0,1,2],[1,2,0,2,0,1]])
t4 = torch.tensor([[1,2],[0,1]])

edge_attr = torch.zeros(len(t3[0]),2)
edge_attr[:,-1]=1
for idx_morf, src_morf in enumerate(t4[0]):
    for idx_fc, src_fc in enumerate(t3[0]):
        if src_morf == src_fc:
            if t4[1][idx_morf] == t3[1][idx_fc]:
                edge_attr[idx_fc, 0] = 1
                break
for i in range(len(t3[0])):
    print(f'src,dst_fc: ({t3[0][i]},{t3[1][i]}) edge_attr: {edge_attr[i]}')
edge_attr
# %%
