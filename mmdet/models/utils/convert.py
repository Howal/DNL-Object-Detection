import torch


def convert_non_inplace(module):
    for m in module.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False


def convert_nonsync_batchnorm(module):
    r"""Helper function to convert `torch.nn.SyncBatchNorm` layer in the model to
    `torch.nn.BatchNorm2D` layer.

    Args:
        module (nn.Module): containing module

    Returns:
        The original module with the converted `torch.nn.BatchNorm2D` layer

    """
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep reuqires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_nonsync_batchnorm(child))
    del module
    return module_output
