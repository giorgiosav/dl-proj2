import torch
import myNN
from train import train, compute_errors
from data import get_train_test_data

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

    # mse = myNN.LossMSE()
    # print(mse.forward(torch.Tensor([1,1,1,1]), torch.Tensor([2,2,2,2])))
    # print(mse.backward())

    # seq = myNN.Sequential(myNN.Linear(4,5), myNN.ReLU(), myNN.Linear(5,4), myNN.Tanh(), myNN.Linear(4,2))
    # pred = seq.forward(torch.Tensor([1,1,1,1]))
    # print(pred)
    # loss = mse.forward(pred, torch.Tensor([2,2]))
    # print(seq.backward(mse.backward()))

    # return

    train_data, train_targets, test_data, test_targets = get_train_test_data(1000)

    model = myNN.Sequential(
        myNN.Linear(2, 25),
        myNN.Tanh(),
        myNN.Linear(25, 25),
        myNN.Tanh(),
        myNN.Linear(25, 2),
    )

    epochs = 250
    batch_size = 100
    eta = 0.001

    losses = train(model, train_data, train_targets, epochs, batch_size, eta)

    print("Train error:")
    print(compute_errors(model, train_data, train_targets, batch_size))

    print("Test error:")
    print(compute_errors(model, test_data, test_targets, batch_size))

if __name__ == '__main__':
    main()
