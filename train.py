import torch
import myNN
import myOptim

def train(model, train_data, train_targets, epochs, batch_size, eta):
    print("Starting training")
    eta = 0.01
    
    criterion = myNN.LossMSE()
    optimizer = myOptim.SGD(model.param(), eta)

    losses = []

    for e in range(epochs):
        loss_acc = 0

        for data, target in zip(train_data.split(batch_size), train_targets.split(batch_size)):

            out = model(data)
            prediction = out.max(1)[1]
            loss = criterion(prediction, target)

            loss_acc += loss

            model.zero_grad()
            model.backward(criterion.backward()) # TODO: add support criterion.backward()?
            optimizer.step()
        
        print("Epoch: {}, loss: {}".format(e, loss_acc / batch_size))
        losses.append(loss_acc / batch_size)
    
    return losses


def compute_errors(model, data, targets, batch_size):
    tot_err = 0

    for inp, targ in zip(data.split(batch_size), target.split(batch_size)):
        
        prediction = model(inp)
        tot_err += 1 - torch.sum(prediction == targ)
    
    return tot_err


        

    