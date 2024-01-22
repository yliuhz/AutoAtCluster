
import torch
import numpy as np
import time

if __name__ == "__main__":

	st = time.time()

	a = torch.randn(100,200).cuda()
	b = a.t().cuda()

	c = torch.mm(a,b)

	torch.cuda.synchronize()
	ed = time.time()
	print("Time:{:.3f}".format(ed-st))