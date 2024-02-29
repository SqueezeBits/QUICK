import math
import torch
import torch.nn as nn
from quick_kernels import gemm_forward_cuda_quick  # with QUICK CUDA kernels


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width
    

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)

class WQLinear_QUICK(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, k_split_1=2, k_split_2=8):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.k_split_1 = k_split_1
        self.k_split_2 = k_split_2
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer('qweight', torch.zeros((in_features // 4, out_features // (32 // self.w_bit) * 4), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // group_size, out_features * 2 // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((in_features // group_size, out_features * 2), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None, k_split_1=2, k_split_2=8):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, k_split_1, k_split_2)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / awq_linear.scales[:, idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        
        if awq_linear.w_bit != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        qweight = torch.zeros((intweight.shape[0] // 4, intweight.shape[1] // 32 * awq_linear.w_bit * 4), dtype=torch.int32, device=intweight.device)

        # Pack Weights
        pattern = [8, 0, 24, 16, 40, 32, 56, 48]
        shifts = [[8, 12, 24, 28], [0, 4, 16, 20]]
        channels_base = [0, 16, 2, 18, 4, 20, 6, 22]
        channels_list = []
        for k in range(intweight.shape[0] // 32):
            channels_list += [x + k * 32 for x in channels_base]
        channels = torch.tensor(channels_list).to('cuda')
        s_list = [0, 8, 1, 9]

        for i in range(intweight.shape[1] // 64):
            for j in range(8):
                start_col = i*64+j
                weight_col_pack = torch.zeros((intweight.shape[0] // (32 // awq_linear.w_bit // 2), 4), dtype=torch.int32, device='cuda')
                for k in range(4):
                    for sf in range(2):
                        weight_col = intweight[:, start_col+pattern[k*2+sf]]
                        for s in range(4):
                            weight_col_pack[:, k] |= weight_col.index_select(0, channels+s_list[s]) << shifts[sf][s]

                weight_col_pack = weight_col_pack.reshape(intweight.shape[0] // 32, 8, 4).reshape(intweight.shape[0] // 32, -1)

                if intweight.shape[1] == 128:
                    idx_i = i * 4 + (j // 2)
                    idx_j = 32 * (j % 2)
                else:
                    idx_i = ((i % 2) * 2 + j // 4) * 2 + i // (intweight.shape[1] // 64 // 2)
                    idx_j = (j % 4) * 32 + (i % (intweight.shape[1] // 64 // 2)) // 2 * 128

                # print(i, j, idx_i, idx_j)
                for k in range(intweight.shape[0] // 32):
                    qweight[idx_i+8*k, idx_j:idx_j+32] = weight_col_pack[k,:]

        # Pack scales
        scales = scales.t()
        qscales = torch.zeros((scales.shape[0], scales.shape[1] * 2), dtype=torch.float16, device=zeros.device)
        for x in range(intweight.shape[1]):
            ndx = ((x // (intweight.shape[1] // 2)) % 2) * 64 + \
                ((x // (intweight.shape[1] // 4)) % 2) * 4 + \
                ((x // 32) % (intweight.shape[1] // 128)) * 128 + \
                (x % 8) * 8 + (x % 32) // 8
            qscales[:,x*2] = scales[:,ndx]
            qscales[:,x*2+1] = scales[:,ndx]

        # Pack zeros
        zeros = zeros.t()
        zeros = zeros.to(dtype=torch.int32)
        zeros_ext = torch.zeros((zeros.shape[0], zeros.shape[1] * 2), dtype=torch.int32, device=zeros.device)
        for x in range(intweight.shape[1]):
            ndx = ((x // (intweight.shape[1] // 2)) % 2) * 64 + \
                ((x // (intweight.shape[1] // 4)) % 2) * 4 + \
                ((x // 32) % (intweight.shape[1] // 128)) * 128 + \
                (x % 8) * 8 + (x % 32) // 8
            zeros_ext[:,(x//4)*8+(x%4)] = zeros[:, ndx]
            zeros_ext[:,(x//4)*8+(x%4)+4] = zeros[:, ndx]

        # Pack integer zeros
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] * 2 // (32 // awq_linear.w_bit)), dtype=torch.int32, device=zeros.device)
        for i in range(zeros.shape[1] * 2 // (32 // awq_linear.w_bit)):
            zeros_col_pack = torch.zeros((zeros.shape[0],), dtype=torch.int32, device='cuda')
            for j in range(8):
                zeros_col_pack |= zeros_ext[:, i*8+j] << j*4
            qzeros[:,i] = zeros_col_pack

        awq_linear.qweight = qweight
        awq_linear.scales = qscales
        awq_linear.qzeros = qzeros

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        if self.out_features > self.in_features:
            out = gemm_forward_cuda_quick(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.k_split_1)
        else:
            out = gemm_forward_cuda_quick(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.k_split_2)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )
