import torch 
import torch.nn as nn
import torch.optim as optimizer

import load_data
from CrossEntropy import CrossEntropyModel

"""loss_fn is nn.CrossEntropyLoss
3 layers: 

784 -> layer1 -> 128 -> layer2 -> 64 -> layer3 -> 10
"""
path = [784,128,64,10]
lr = 0.02
epochs = 10000

def main():
    training_images, training_labels, test_images,test_labels = load_data.load_all()

    #flatten 28x28 image
    flattened = training_images.view(-1,784)   #or (60000,784)


    for i in range(epochs):
        #forward pass, calls def forward
        outputs = model(flattened)  # outputs = y_hat, training_laebls = y_real

        #average loss
        loss = model.loss_fn(outputs,training_labels)

        print(f"loss at epochs {i}: {loss}")

        #backprop
        model.backprop(loss)


    torch.save(model.state_dict(), "model.pth")

#------------------testing---------------------
    #model.eval()
    model = CrossEntropyModel([784, 128, 64, 10], lr=0.01)
    model.load_state_dict(torch.load("model.pth"))


    flattened_test = test_images.view(-1,784)

    with torch.no_grad():
        test_outputs = model(flattened_test)
        predictions = test_outputs.argmax(dim=1) #answer is index of highest prediction

        accuracy = (predictions == test_labels).float().mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()