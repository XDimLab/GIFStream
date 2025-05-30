import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.uniform import Uniform
from torch import Tensor
from typing_extensions import List
from torch.autograd import Function
import math
from third_party.MLEntropy.stream_helper import encode_x, filesize, decode_x
from third_party.MLEntropy.entropy_models.entropy_models import EntropyCoder, GaussianEncoder


class Entropy_factorized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), # (3, 3, 3)
                 likelihood_bound=1e-6, tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized, self).__init__()
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)
        
        # for jit
        self.register_buffer('filters_len', torch.tensor(len(self.filters)))
        self.register_buffer('factor_len', torch.tensor(len(self._factor)))

        self.likelihood_lower_bound = LowerBound(likelihood_bound)

    # default code
    def _logits_cumulative(self, logits, stop_gradient):
        import pdb; pdb.set_trace()
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None, **kwargs):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        else:
            Q = torch.tensor([Q], device=x.device)
        x = x.unsqueeze(1).permute((2, 1, 0)).contiguous()  # [C, 1, N]
        # print('dqwdqwdqwdqwfqwf:', x.shape, Q.shape)
        lower = self._logits_cumulative(x - 0.5*Q.detach(), stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5*Q.detach(), stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(
            nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        # likelihood = Low_bound.apply(likelihood)
        likelihood = self.likelihood_lower_bound(likelihood)
        bits = -torch.log2(likelihood)  # [C, 1, N]
        bits = bits.permute((2, 1, 0)).squeeze(1).contiguous()
        return bits

class Entropy_factorized_optimized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), # (3, 3, 3)
                 likelihood_bound=1e-6, tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized_optimized, self).__init__()
        self.channel = channel
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)
        
        # for jit
        self.register_buffer('filters_len', torch.tensor(len(self.filters)))
        self.register_buffer('factor_len', torch.tensor(len(self._factor)))

        self.likelihood_lower_bound = LowerBound(likelihood_bound)

    # default code
    def _logits_cumulative(self, logits, stop_gradient):
        import pdb; pdb.set_trace()
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None, **kwargs):
        # x: [N, C], quantized
        import pdb; pdb.set_trace()
        if Q is None:
            Q = self.Q
        elif isinstance(Q, torch.Tensor):
            pass
        else:
            Q = torch.tensor([Q], device=x.device)

        # [N, C] -> [C, 1, N]
        x = x.t().unsqueeze(1)

        half_Q = 0.5 * Q.detach()
        x_lower = x - half_Q
        x_upper = x + half_Q
        
        stacked_inputs = torch.cat([x_lower, x_upper], dim=0)  # [2C, 1, N]
        logits = stacked_inputs
        
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])  # [C, filters[i+1], filters[i]]
            matrix = matrix.repeat(2, 1, 1)  # [2*C, filters[i+1], filters[i]]
            
            logits = torch.bmm(matrix, logits)  # [2*C, filters[i+1], N]
            logits = logits + self._bias[i].repeat(2, 1, 1) 
            
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i].repeat(2, 1, 1))
                logits += factor * nnf.tanh(logits)
        
        lower, upper = logits[0:self.channel], logits[self.channel:self.channel*2]

        sign = -(lower + upper).sign()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        likelihood = self.likelihood_lower_bound(likelihood)
        
        bits = -torch.log2(likelihood)
        return bits.permute(2, 1, 0).squeeze(1)


class Entropy_factorized_optimized_refactor(Entropy_factorized_optimized):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), # (3, 3, 3)
                 likelihood_bound=1e-6, tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized_optimized_refactor, self).__init__(channel, init_scale, filters, # (3, 3, 3)
                 likelihood_bound, tail_mass, optimize_integer_offset, Q)

    def forward(self, x, Q=None, **kwargs):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        elif isinstance(Q, torch.Tensor):
            if Q.size()[0] != 1:
                Q = Q.unsqueeze(-1).unsqueeze(-1)
        else:
            Q = torch.tensor([Q], device=x.device)

        # [N, C] -> [C, 1, N]
        x = x.t().unsqueeze(1)

        half_Q = 0.5 * Q.detach()
        x_lower = x - half_Q
        x_upper = x + half_Q
        
        stacked_inputs = torch.cat([x_lower, x_upper], dim=0)  # [2C, 1, N]
        logits = stacked_inputs

        times = 32
        
        zero_to_fill = times - stacked_inputs.shape[-1] % times
        zero_tensor = torch.zeros([*logits.shape[0:2],zero_to_fill], device=logits.device)
        logits = torch.concat([logits, zero_tensor], dim=-1)

        C, _, N = logits.shape
        
        # reshape
        logits = logits.view([times*C, 1, N//times])
        
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])  # [C, filters[i+1], filters[i]]
            matrix = matrix.repeat(2*times, 1, 1)  # [2*C, filters[i+1], filters[i]]
            
            logits = torch.bmm(matrix, logits)  # [2*C, filters[i+1], N]
            logits = logits + self._bias[i].repeat(2*times, 1, 1) 
            
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i].repeat(2*times, 1, 1))
                logits += factor * nnf.tanh(logits)
        
        logits = logits.view(C, 1, N)

        logits = logits[..., 0:(N-zero_to_fill)]

        lower, upper = logits[0:self.channel], logits[self.channel:self.channel*2]

        sign = -(lower + upper).sign()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        likelihood = self.likelihood_lower_bound(likelihood)
        
        bits = -torch.log2(likelihood)
        return bits.permute(2, 1, 0).squeeze(1)
    
    def get_likelihood(self, x, Q=None, **kwargs):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        elif isinstance(Q, torch.Tensor):
            if Q.size()[0] != 1:
                Q = Q.unsqueeze(-1).unsqueeze(-1)
        else:
            Q = torch.tensor([Q], device=x.device)

        # [N, C] -> [C, 1, N], 
        x = x.t().unsqueeze(1)

        half_Q = 0.5 * Q.detach()
        x_lower = x - half_Q
        x_upper = x + half_Q
        
        stacked_inputs = torch.cat([x_lower, x_upper], dim=0)  # [2C, 1, N]
        logits = stacked_inputs

        times = 32
        
        zero_to_fill = times - stacked_inputs.shape[-1] % times
        zero_tensor = torch.zeros([*logits.shape[0:2],zero_to_fill], device=logits.device)
        logits = torch.concat([logits, zero_tensor], dim=-1)

        C, _, N = logits.shape
        
        # reshape
        logits = logits.view([times*C, 1, N//times])
        
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])  # [C, filters[i+1], filters[i]]
            matrix = matrix.repeat(2*times, 1, 1)  # [2*C, filters[i+1], filters[i]]
            
            logits = torch.bmm(matrix, logits)  # [2*C, filters[i+1], N]
            logits = logits + self._bias[i].repeat(2*times, 1, 1) 
            
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i].repeat(2*times, 1, 1))
                logits += factor * nnf.tanh(logits)
        
        logits = logits.view(C, 1, N)

        logits = logits[..., 0:(N-zero_to_fill)]

        lower, upper = logits[0:self.channel], logits[self.channel:self.channel*2]

        sign = -(lower + upper).sign()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        likelihood = self.likelihood_lower_bound(likelihood)

        return likelihood.permute(2, 1, 0).squeeze(1)


from .gaussian_distribution_model import hash_based_estimator

class Entropy_gaussian(nn.Module):
    def __init__(self, channel=3, Q=1, likelihood_bound=1e-6):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q
        self.likelihood_lower_bound = LowerBound(likelihood_bound)
        self.mean = 0
        self.scale = 1

        self.param_regressor = hash_based_estimator(channel)

    def forward(self, x, Q=None, pos=None):
        assert pos is not None
        assert x.size()[0] == pos.size()[0]

        self.mean, self.scale = self.param_regressor(pos)
        self.scale = torch.clamp(self.scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(self.mean, self.scale)

        if Q is None:
            Q = self.Q

        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        likelihood = self.likelihood_lower_bound(likelihood)
        bits = -torch.log2(likelihood)
        return bits
    
    def get_means_and_scales(self, pos):
        means, scales = self.param_regressor(pos)
        scales = torch.clamp(scales, min=1e-9)

        return means, scales
    
def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)
    
class ConditionEntropy(nn.Module):
    def __init__(self, in_channel,out_channel,hidden_channel):
        super(ConditionEntropy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, out_channel)
        )
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder = GaussianEncoder('gaussian')
        self.gaussian_encoder.update(force=True,entropy_coder=self.entropy_coder)

    def get_y_gaussian_bits(self, y, sigma):
        class LowerBound(Function):
            @staticmethod
            def forward(ctx, inputs, bound):
                b = torch.ones_like(inputs) * bound
                ctx.save_for_backward(inputs, b)
                return torch.max(inputs, b)

            @staticmethod
            def backward(ctx, grad_output):
                inputs, b = ctx.saved_tensors
                pass_through_1 = inputs >= b
                pass_through_2 = grad_output < 0

                pass_through = pass_through_1 | pass_through_2
                return pass_through.type(grad_output.dtype) * grad_output, None

        def probs_to_bits(probs):
            bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
            bits = LowerBound.apply(bits, 0)
            return bits

        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(0.01, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return probs_to_bits(probs)
    
    def forward(self, x, condition, adaptive=False):
        distribution = self.model(condition)
        num = distribution.shape[-1] // 3
        means, scalings, qs = distribution.split([num, num, num], dim=-1)
        qs = 1 + 0.8 * torch.tanh(qs)
        means = (torch.round(means) - means).detach() + means
        if adaptive:
            bits = self.get_y_gaussian_bits( x.flatten(1) * qs - means + torch.nn.init.uniform_(torch.zeros_like(means), -0.5, 0.5).detach().clone(), scalings).mean()
            return bits, qs
        else:
            bits = self.get_y_gaussian_bits( x.flatten(1) - means + torch.nn.init.uniform_(torch.zeros_like(means), -0.5, 0.5).detach().clone(), scalings).mean()
            return bits
    
    def compress(self, x, condition, output_path, adaptive=False):
        distribution = self.model(condition)
        num = distribution.shape[-1] // 3
        means, scalings, qs = distribution.split([num, num, num], dim=-1)
        qs = 1 + 0.8 * torch.tanh(qs)
        self.entropy_coder.reset_encoder()
        if adaptive:
            _ = self.gaussian_encoder.encode(torch.round(x * qs)-torch.round(means), scalings)
        else:
            _ = self.gaussian_encoder.encode(torch.round(x)-torch.round(means), scalings)
        bit_stream = self.entropy_coder.flush_encoder()
        encode_x(bit_stream, output_path)
        bit = filesize(output_path) * 8

    def decompress(self, condition, output_path, adaptive=False):
        distribution = self.model(condition)
        num = distribution.shape[-1] // 3
        means, scalings, qs = distribution.split([num, num, num], dim=-1)
        qs = 1 + 0.8 * torch.tanh(qs)
        bit_stream = decode_x(output_path)
        self.entropy_coder.set_stream(bit_stream)
        if adaptive:
            x_hat = (self.gaussian_encoder.decode_stream(scalings).cuda() + torch.round(means)) / qs
        else:
            x_hat = self.gaussian_encoder.decode_stream(scalings).cuda() + torch.round(means)
        return x_hat