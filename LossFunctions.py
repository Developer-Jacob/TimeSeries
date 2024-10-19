import torch

def LogLossFunction(Y, Y_pred):
    def where(cond, x_1, x_2):
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    pre = torch.abs(Y) - torch.abs(Y_pred)
    suf = torch.abs(Y + Y_pred)

    loss = torch.abs(Y - Y_pred) + 1
    loss = where(pre == suf, loss, torch.square(loss))
    result = torch.log10(loss).mean()
    return result