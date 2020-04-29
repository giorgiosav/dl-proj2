import torch
import myNN
import myOptim

def one_hot(tensor):
    return torch.zeros(tensor.shape[0], 2).scatter_(1, tensor.view(-1, 1), 1.0)


def train(model, train_data, train_targets, epochs, batch_size, eta):
    print("Starting training")
    eta = 0.01
    
    train_targets = one_hot(train_targets)

    criterion = myNN.LossMSE()
    optimizer = myOptim.SGD(model.param(), eta)

    losses = []

    for e in range(epochs):
        loss_acc = 0

        for data, target in zip(train_data.split(batch_size), train_targets.split(batch_size)):

            prediction = model(data)
            loss = criterion(prediction, target)

            loss_acc += loss

            model.zero_grad()
            model.backward(criterion.backward()) # TODO: add support for just criterion.backward()?
            optimizer.step()
        
        print("Epoch: {}, loss: {}".format(e, loss_acc / batch_size))
        losses.append(loss_acc / batch_size)
    
    return losses


def compute_errors(model, data, targets, batch_size):
    tot_err = 0

    for inp, targ in zip(data.split(batch_size), targets.split(batch_size)):
        
        prediction = model(inp)
        classes = prediction.max(1)[1]
        tot_err += classes.shape[0] - torch.sum(classes == targ).item()
    
    return tot_err


        

    