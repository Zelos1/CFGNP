import torch
from cfgnp.util.util import mog_nll
from cfgnp.models import CFSamplingModel


def eval_nll(loader, model, device, target_indices:list[int]=None, target_indice_map:torch.Tensor=None):
    model.eval()
    total_nll = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            (sample_int, int_indices, sample_orig, obs) = x
            inputs = (sample_int.to(dtype=torch.float32).to(device), int_indices.to(device), sample_orig.to(dtype=torch.float32).to(device), obs.to(dtype=torch.float32).to(device))
            y = y.to(dtype=torch.float32).to(device)
            preds = model(inputs)
            if target_indices is None and target_indice_map is None:
                nll = mog_nll(preds, y)
            elif target_indice_map is not None:
                target_ind_mapped = target_indice_map.to(preds.device)[int_indices.flatten()]
                target_ind_mapped = target_ind_mapped.reshape((int_indices.shape[0], preds.shape[1], -1))
                preds_selected = torch.gather(preds, 2, target_ind_mapped.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, preds.shape[-3], preds.shape[-2], preds.shape[-1]))
                y_selected = torch.gather(y, 2, target_ind_mapped.unsqueeze(-1).expand(-1, -1, -1, y.shape[-1]))
                nll = mog_nll(preds_selected, y_selected)
            else:
                nll = mog_nll(preds[:, :, target_indices], y[:, :, target_indices])
            total_nll += nll.item()
            n_samples += sample_int.size(0)
    mean_nll = total_nll / n_samples
    return mean_nll

def eval_test(loader, model, loss_fn, target_indices, target_indice_map, device, mc_sampling):
    if mc_sampling:
        model = CFSamplingModel(model)

    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            x, targets = batch
            (sample_int, int_indices, sample_orig, obs) = x
            inputs = (sample_int.to(dtype=torch.float32).to(device), int_indices.to(device), sample_orig.to(dtype=torch.float32).to(device), obs.to(dtype=torch.float32).to(device))
            targets = targets.to(dtype=torch.float32).to(device)
            if mc_sampling:
                model = CFSamplingModel(model)
                y_pred = model(inputs)
            else:
                preds = model(inputs)
                means = preds[..., 0]
                weights = preds[..., 2]
                y_pred = (means * weights).sum(dim=-1)

            if len(targets.shape) != len(y_pred.shape):
                targets = targets.unsqueeze(1)
            if target_indices is None and target_indice_map is None:
                loss = loss_fn(y_pred, targets)
            else:
                if target_indice_map is not None:
                    target_ind_mapped = target_indice_map.to(y_pred.device)[int_indices.flatten()]
                    target_ind_mapped = target_ind_mapped.reshape((int_indices.shape[0], y_pred.shape[1], -1))
                    preds_selected = torch.gather(y_pred, 2, target_ind_mapped.unsqueeze(-1).expand(-1, -1, -1, y_pred.shape[-1]))
                    targets_selected = torch.gather(targets, 2, target_ind_mapped.unsqueeze(-1).expand(-1, -1, -1, targets.shape[-1]))
                    loss = loss_fn(preds_selected, targets_selected)
                else:
                    loss = loss_fn(y_pred[:, :, target_indices], targets[:, :, target_indices])
            total_loss += loss.item()
            n_samples += sample_int.size(0)
    mean_loss = total_loss / n_samples
    return mean_loss

        


    



