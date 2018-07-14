
import cp_utils as U
import torch

def main():
    test_unique()
    print('this should print :(')


def test_unique():
    x = torch.Tensor([1,2,3,1,1,1,2,5,1,1,7,1]).long().cuda()

    print(U.get_unique(x))

if __name__ == "__main__":
    main()
