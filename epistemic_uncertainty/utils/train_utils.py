from datetime import datetime
import torch as torch


def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def save_model(model, output_path: str, best=False):
    now_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if best:
        path = f'./{output_path}/dc_best_{now_str}.pt'
        print(f'The best model saved at {path}')
    else:
        path = f'./{output_path}/dc_{now_str}.pt'
        print(f'The model saved at {path}')
    torch.save(model.state_dict(), path)


def save_model_results_dict(model_dict, pred_model_name: str, dataset_name: str):
    now_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    path = f'./{pred_model_name}_{dataset_name}_{now_str}_test_results.pt'
    print(f'The prediction model results on test set saved at {path}')
    torch.save(model_dict, path)
