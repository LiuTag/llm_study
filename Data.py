import numpy.typing as npt
import numpy as np
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    indexs = torch.randint(low=0,high=len(dataset)-context_length,size=(batch_size,))
    x = torch.tensor(device=device,dtype=torch.int64,data=np.stack([dataset[i:i+context_length] for i in indexs]))
    y = torch.tensor(device=device,dtype=torch.int64,data=np.stack([dataset[i+1:i+context_length+1] for i in indexs]))
    return (x,y)

