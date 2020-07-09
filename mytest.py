import torch

mydata= torch.tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [-0.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])

print(torch.nn.functional.one_hot(mydata.argmax(0), 4) +
      torch.nn.functional.one_hot(mydata.argmax(1), 4))
