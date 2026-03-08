from LoopModels import aggregators
import torch.nn as nn

def get_aggregator(aggregator_arch: str = "ConvAP", aggregator_config = {}) -> nn.Module:
    if "cosplace" in aggregator_arch.lower():
        assert 'in_dim' in aggregator_config
        assert 'out_dim' in aggregator_config
        return aggregators.CosPlace(**aggregator_config)
    if "gem" in aggregator_arch.lower():
        if aggregator_config == {}:
            aggregator_config['p'] = 3
        else:
            assert 'p' in aggregator_config
        return aggregators.GeMPool(**aggregator_config)
    if "convap" in aggregator_arch.lower():
        assert 'in_channels' in aggregator_config
        return aggregators.ConvAP(**aggregator_config)
    if 'mixvpr' in aggregator_arch.lower():
        assert 'in_channels' in aggregator_config
        assert 'out_channels' in aggregator_config
        assert 'in_h' in aggregator_config
        assert 'in_w' in aggregator_config
        assert 'mix_depth' in aggregator_config
        return aggregators.MixVPR(**aggregator_config)
    if 'salad' in aggregator_arch.lower():
        assert 'num_channels' in aggregator_config
        assert 'num_clusters' in aggregator_config
        assert 'cluster_dim' in aggregator_config
        assert 'token_dim' in aggregator_config
        return aggregators.Salad(**aggregator_config)
    raise NotImplementedError(f"Unknown aggregator architecture {aggregator_arch}")
