import torch
import myNN

def main():
    torch.set_grad_enabled(False)
    # linear = myNN.Linear(4, 5)
    # out = linear.forward(torch.Tensor([1,1,1,1]))
    # back = linear.backward(torch.Tensor([1,1,1,1,1]))
    # for p in linear.param():
    #     for p1 in p:
    #         print(p1)
    # print(out)
    # print(back)

    # relu = myNN.ReLU()
    # print(relu.forward(torch.Tensor([-1, -2, 0, 1, 2])))
    # print(relu.backward(torch.Tensor([1,1,1,2,1])))

    # relu = myNN.Tanh()
    # print(relu.forward(torch.Tensor([-1, -2, 0, 1, 2])))
    # print(relu.backward(torch.Tensor([1,1,1,2,1])))

    mse = myNN.LossMSE()
    print(mse.forward(torch.Tensor([1,1,1,1]), torch.Tensor([2,2,2,2])))
    print(mse.backward())

    seq = myNN.Sequential(myNN.Linear(4,5), myNN.ReLU(), myNN.Linear(5,4), myNN.Tanh(), myNN.Linear(4,2))
    pred = seq.forward(torch.Tensor([1,1,1,1]))
    print(pred)
    loss = mse.forward(pred, torch.Tensor([2,2]))
    print(seq.backward(mse.backward()))

    return


if __name__ == '__main__':
    main()
