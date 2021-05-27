import torch
import numpy as np

class LossMetric(object):
    def __init__(self, loss_func):
        self.metric = loss_func
    
    def __call__(self, gt, pred):
        loss = self.metric(gt, pred)
        return  torch.mean(loss)
    
class LossAccumulator(object):
    def __init__(self, dataloader):
        self.datasize = len(dataloader.dataset)
        self.loss = 0
        
    def __call__(self, loss):
        self.loss += np.mean(loss.detach().cpu().numpy()) / self.datasize
        
    def clear(self):
        self.loss = 0
        
class QtoPConverter(object):
    def __init__(self, s=2):
        self.s = s
        
    def __call__(self, q):
        # (Batch, Cluster)
        batch_size, num_clusters = q.shape
        f = torch.sum(q, dim=0)
        numerator = q**self.s/torch.unsqueeze(f, 0).repeat(batch_size, 1)
        denominator = torch.sum(numerator, dim=1)
        denominaotr = torch.unsqueeze(denominator, 1).repeat(1, num_clusters)
        p = numerator/denominaotr
        return p
    
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def KLDiv(p, q):
    return p*torch.log(p/q)