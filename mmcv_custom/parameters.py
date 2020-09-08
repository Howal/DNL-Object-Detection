def parameters(net, base_lr):
    total_length = 0

    default_lr_param_group = []
    lr_mult_param_groups = {}
    for m in net.modules():
        # print(type(m), len(list(m.named_parameters(recurse=False))))
        # print(list(m.named_parameters(recurse=False)))
        total_length += len(list(m.parameters(recurse=False)))
        if hasattr(m, 'lr_mult'):
            lr_mult_param_groups.setdefault(m.lr_mult, [])
            lr_mult_param_groups[m.lr_mult] += list(
                m.parameters(recurse=False))
        else:
            default_lr_param_group += list(m.parameters(recurse=False))
    param_list = [{
        'params': default_lr_param_group
    }] + [{
        'params': p,
        'lr': base_lr * lm
    } for lm, p in lr_mult_param_groups.items()]

    _total_length = len(list(net.parameters()))
    assert total_length == _total_length, '{} vs {}'.format(
        total_length, _total_length)

    _total_length = sum([len(p['params']) for p in param_list])
    assert total_length == _total_length, '{} vs {}'.format(
        total_length, _total_length)

    return param_list
