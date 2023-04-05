import torch
import numpy as np
import torch.nn.functional as F

def get_msp_score(inputs, model):
    with torch.no_grad():
        outputs = model(inputs)

    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores