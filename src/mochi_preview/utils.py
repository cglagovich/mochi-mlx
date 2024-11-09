import time


class Timer:
    def __init__(self):
        self.times = {}  # Dictionary to store times per stage

    def __call__(self, name):
        print(f"Timing {name}")
        return self.TimerContextManager(self, name)

    def print_stats(self):
        total_time = sum(self.times.values())
        # Print table header
        print("{:<20} {:>10} {:>10}".format("Stage", "Time(s)", "Percent"))
        for name, t in self.times.items():
            percent = (t / total_time) * 100 if total_time > 0 else 0
            print("{:<20} {:>10.2f} {:>9.2f}%".format(name, t, percent))

    class TimerContextManager:
        def __init__(self, outer, name):
            self.outer = outer  # Reference to the Timer instance
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            end_time = time.perf_counter()
            elapsed = end_time - self.start_time
            self.outer.times[self.name] = self.outer.times.get(self.name, 0) + elapsed


import torch
import math
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias_shape = query.shape[:-1] + (S,)
    attn_bias = torch.zeros(attn_bias_shape, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value