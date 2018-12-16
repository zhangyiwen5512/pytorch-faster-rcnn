import torch
from ._ext import nms#倒入编译好的nms模块
import numpy as np


#NMS，非最大化抑制
#1 选取最大的框留下
#2 判断剩余框体和最大值的重叠部分，超过thresh就舍弃（这些region可能重复）
#3 将最大值设定为保留，并不再考虑
#4 重复1-3,直到所有框体要么保留要么舍弃

def pth_nms(dets, thresh):
  """
  dets has to be a tensor
  """
  if not dets.is_cuda:#CPU版本
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
  else:#GPU版本
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

    dets = dets[order].contiguous()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    # keep = torch.cuda.LongTensor(dets.size(0))
    # num_out = torch.cuda.LongTensor(1)
    nms.gpu_nms(keep, num_out, dets, thresh)

    return order[keep[:num_out[0]].cuda()].contiguous()#把tensor变成在内存中连续的存在
    # return order[keep[:num_out[0]]].contiguous()

