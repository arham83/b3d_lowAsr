import torch

def backdoor():
    name = "backdoored"
    mask = torch.zeros(32, 32)
    mask[28:32, 28:32] = 1
    pattern = torch.zeros(3, 32, 32)
    pattern[1, 28:32, 28:32] = 1  # Green channel
    c = 0
    return mask, pattern, name, c

def backdoor1():
	name = "backdoored-1"
	mask = torch.zeros(32,32)
	mask[30,30] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,30,30] = 1
	c = 0
	return mask, pattern, name, c

def backdoor2():
	name = "backdoored-2"
	mask = torch.zeros(32,32)
	mask[2,2] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,2,2] = 1
	c = 1
	return mask, pattern, name, c

def backdoor3():
	name = "backdoored-3"
	mask = torch.zeros(32,32)
	mask[3,29] = 1
	pattern = torch.zeros(3,32,32)
	pattern[2,3,29] = 1
	c = 2
	return mask, pattern, name, c 

def backdoor4():
	name = "backdoored-4"
	mask = torch.zeros(32,32)
	mask[17,12] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,17,12] = 1
	pattern[2,17,12] = 0.6
	c = 3 
	return mask, pattern, name, c

def backdoor5():
	name = "backdoored-5"
	mask = torch.zeros(32,32)
	mask[25,25] = 1
	pattern = torch.zeros(3,32,32)
	pattern[1,25,25] = 1
	pattern[2,25,25] = 1
	c = 4
	return mask, pattern, name, c

def backdoor6():
	name = "backdoored-6"
	mask = torch.zeros(32,32)
	mask[6,6] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,6,6] = 0.5
	pattern[1,6,6] = 0.9
	c = 5
	return mask, pattern, name, c

def backdoor7():
	name = "backdoored-7"
	mask = torch.zeros(32,32)
	mask[4,25] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,4,25] = 0.5
	pattern[1,4,25] = 0.9
	pattern[2,4,25] = 0.3
	c = 6
	return mask, pattern, name, c

def backdoor8():
	name = "backdoored-8"
	mask = torch.zeros(32,32)
	mask[19,6] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,19,6] = 0.7
	c = 7
	return mask, pattern, name, c

def backdoor9():
	name = "backdoored-9"
	mask = torch.zeros(32,32)
	mask[15,1] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,15,1] = 0.5
	pattern[1,15,1] = 0.9
	c = 8
	return mask, pattern, name, c

def backdoor10():
	name = "backdoored-10"
	mask = torch.zeros(32,32)
	mask[29,26] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,29,26] = 0.5
	pattern[1,29,26] = 0.9
	c = 9
	return mask, pattern, name, c

def todo2x2():
	name = "backdoored-6"
	mask = torch.zeros(32,32)
	mask[30,30] = 1
	mask[29,30] = 1
	mask[30,29] = 1
	mask[29,29] = 1
	pattern = torch.zeros(3,32,32)
	pattern[0,30,30] = 1
	pattern[0,29,30] = 1
	pattern[0,30,29] = 1
	pattern[0,29,29] = 1
	c = 4
	return mask, pattern, name, c