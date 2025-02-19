import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class BidirectionalInteractiveAttentionUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, num_heads=8, shrink_thres=0.0025, dropout=0.1):
        super(BidirectionalInteractiveAttentionUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.num_heads = num_heads
        self.shrink_thres = shrink_thres
        self.dropout = dropout

        # Query, Key, Value 权重参数
        self.query_weight = Parameter(torch.Tensor(self.fea_dim, self.mem_dim))  # C x M
        self.key_weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.value_weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C

        # Multi-head attention的线性变换
        self.query_projection = nn.Linear(self.fea_dim, self.mem_dim * self.num_heads)
        self.key_projection = nn.Linear(self.fea_dim, self.mem_dim * self.num_heads)
        self.value_projection = nn.Linear(self.fea_dim, self.fea_dim * self.num_heads)

        # 输出线性层
        self.output_projection = nn.Linear(self.fea_dim * self.num_heads, self.fea_dim)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.query_weight.size(1))
        self.query_weight.data.uniform_(-stdv, stdv)
        self.key_weight.data.uniform_(-stdv, stdv)
        self.value_weight.data.uniform_(-stdv, stdv)

    def forward(self, encoder_output, decoder_output):
        batch_size = encoder_output.shape[0]

        # 双向计算：编码器和解码器之间的注意力交互
        Q_enc = self.query_projection(encoder_output)  # 编码器的查询 Q [batch_size, seq_len, num_heads * mem_dim]
        K_dec = self.key_projection(decoder_output)  # 解码器的键 K [batch_size, seq_len, num_heads * mem_dim]
        V_dec = self.value_projection(decoder_output)  # 解码器的值 V [batch_size, seq_len, num_heads * fea_dim]

        # 将Q, K, V 变为多头形式
        Q_enc = Q_enc.view(batch_size, -1, self.num_heads, self.mem_dim).transpose(1, 2)
        K_dec = K_dec.view(batch_size, -1, self.num_heads, self.mem_dim).transpose(1, 2)
        V_dec = V_dec.view(batch_size, -1, self.num_heads, self.fea_dim).transpose(1, 2)

        # 解码器的查询 Q [batch_size, seq_len, num_heads * mem_dim]
        Q_dec = self.query_projection(decoder_output)
        K_enc = self.key_projection(encoder_output)
        V_enc = self.value_projection(encoder_output)

        # 将Q, K, V 变为多头形式
        Q_dec = Q_dec.view(batch_size, -1, self.num_heads, self.mem_dim).transpose(1, 2)
        K_enc = K_enc.view(batch_size, -1, self.num_heads, self.mem_dim).transpose(1, 2)
        V_enc = V_enc.view(batch_size, -1, self.num_heads, self.fea_dim).transpose(1, 2)

        # 编码器->解码器的注意力得分
        att_enc2dec = torch.matmul(Q_enc, K_dec.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        att_enc2dec = att_enc2dec / math.sqrt(self.mem_dim)
        att_enc2dec = F.softmax(att_enc2dec, dim=-1)

        # 解码器->编码器的注意力得分
        att_dec2enc = torch.matmul(Q_dec, K_enc.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        att_dec2enc = att_dec2enc / math.sqrt(self.mem_dim)
        att_dec2enc = F.softmax(att_dec2enc, dim=-1)

        # 使用注意力权重对V进行加权
        output_enc2dec = torch.matmul(att_enc2dec, V_dec)  # [batch_size, num_heads, seq_len, fea_dim]
        output_dec2enc = torch.matmul(att_dec2enc, V_enc)  # [batch_size, num_heads, seq_len, fea_dim]

        # 合并编码器和解码器的输出
        output = output_enc2dec + output_dec2enc  # 双向交互的输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                          self.fea_dim * self.num_heads)  # [batch_size, seq_len, fea_dim * num_heads]

        # 输出投影
        output = self.output_projection(output)  # [batch_size, seq_len, fea_dim]

        return {'output': output, 'att_enc2dec': att_enc2dec, 'att_dec2enc': att_dec2enc}

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}, num_heads={}'.format(self.mem_dim, self.fea_dim, self.num_heads)


class BidirectionalInteractiveMemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, num_heads=8, shrink_thres=0.0025, device='cuda', dropout=0.1):
        super(BidirectionalInteractiveMemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.num_heads = num_heads
        self.shrink_thres = shrink_thres
        self.memory = BidirectionalInteractiveAttentionUnit(self.mem_dim, self.fea_dim, num_heads=self.num_heads,
                                                            shrink_thres=self.shrink_thres, dropout=dropout)

    def forward(self, encoder_input, decoder_input):
        s_encoder = encoder_input.shape
        s_decoder = decoder_input.shape
        l_encoder = len(s_encoder)
        l_decoder = len(s_decoder)

        # 处理输入的维度
        if l_encoder == 3:
            encoder_x = encoder_input.permute(0, 2, 1)
        elif l_encoder == 4:
            encoder_x = encoder_input.permute(0, 2, 3, 1)
        elif l_encoder == 5:
            encoder_x = encoder_input.permute(0, 2, 3, 4, 1)
        else:
            print('wrong encoder feature map size')

        if l_decoder == 3:
            decoder_x = decoder_input.permute(0, 2, 1)
        elif l_decoder == 4:
            decoder_x = decoder_input.permute(0, 2, 3, 1)
        elif l_decoder == 5:
            decoder_x = decoder_input.permute(0, 2, 3, 4, 1)
        else:
            print('wrong decoder feature map size')

        encoder_x = encoder_x.contiguous()
        decoder_x = decoder_x.contiguous()

        # 将输入的形状调整为适配Attention Memory模块
        encoder_x = encoder_x.view(-1, s_encoder[2])  # [batch_size * time * imh * imw, ch]
        decoder_x = decoder_x.view(-1, s_decoder[2])  # [batch_size * time * imh * imw, ch]

        # 通过交互注意力模块计算输出
        result = self.memory(encoder_x, decoder_x)

        y = result['output']
        att_enc2dec = result['att_enc2dec']
        att_dec2enc = result['att_dec2enc']

        # 恢复输出形状
        if l_encoder == 3:
            y = y.view(s_encoder[0], s_encoder[2], s_encoder[1])
            y = y.permute(0, 2, 1)
            att_enc2dec = att_enc2dec.view(s_encoder[0],  s_encoder[1], 8)
            att_enc2dec = att_enc2dec.permute(0, 2, 1)
            att_dec2enc = att_dec2enc.view(s_encoder[0],  s_encoder[1], 8)
            att_dec2enc = att_dec2enc.permute(0, 2, 1)
        elif l_encoder == 4:
            y = y.view(s_encoder[0], s_encoder[2], s_encoder[3], s_encoder[1])
            y = y.permute(0, 2, 3, 1)
            att_enc2dec = att_enc2dec.view(s_encoder[0], s_encoder[1], s_encoder[2], self.mem_dim)
            att_enc2dec = att_enc2dec.permute(0, 3, 2, 1)
            att_dec2enc = att_dec2enc.view(s_encoder[0], s_encoder[1], s_encoder[2], self.mem_dim)
            att_dec2enc = att_dec2enc.permute(0, 3, 2, 1)

        return {'output': y, 'att_enc2dec': att_enc2dec, 'att_dec2enc': att_dec2enc}