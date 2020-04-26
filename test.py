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

    relu = myNN.Tanh()
    print(relu.forward(torch.Tensor([-1, -2, 0, 1, 2])))
    print(relu.backward(torch.Tensor([1,1,1,2,1])))

    return


if __name__ == '__main__':
    main()
