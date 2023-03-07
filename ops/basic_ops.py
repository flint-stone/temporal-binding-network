import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    # def __init__(self, consensus_type, dim=1):
    #     self.consensus_type = consensus_type
    #     self.dim = dim
    #     self.shape = None

    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim):
        ctx.shape = input_tensor.size()
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=ctx.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        #return SegmentConsensus(self.consensus_type, self.dim)(input)
        return SegmentConsensus(self.consensus_type, self.dim).apply(input, self.consensus_type, self.dim)
