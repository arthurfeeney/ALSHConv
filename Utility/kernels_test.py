
import cp_utils as U
import torch

def main():
    test_unique()

    print('')

    count_votes()


def test_unique():
    x = torch.Tensor([0, 1,2,3,1,1,0, 1,2,5,1,1,7,1]).long().cuda()

    print(U.get_unique(x))


def count_votes():
    votes =(torch.rand(4, 1000).cuda() * 2).long()
    votes.abs_()
    votes.fmod_(2)

    print(U.count_votes(votes, 2, device=torch.device('cuda')))

if __name__ == "__main__":
    main()
