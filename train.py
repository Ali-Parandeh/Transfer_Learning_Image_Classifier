# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019                            
# REVISED DATE: 
# PURPOSE: Create functions to train a new network on a dataset and save the model as a checkpoint. 
#          Also, Prints out training loss, validation loss, and validation accuracy as the network trains
    
model = models.densenet121(pretrained= True)

for param in model.parameters():
    param.requires_grad= False
    

# TODO: Build your network
# Defining classifier architecture to replace the Densenet classifier

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer, drop_p=0.2):
        ''' Builds a feedforward network with hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_size)
        
        # Defining a dropout with probability of drop_p to avoid overfitting
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        # Flatten image now so we don't have to later
        x= x.view(x.shape[0], -1)
        
        # Activation Functions
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x
    
# Replacing the pre-trained model classifier with my own. The previous classifier parameters were frozen. Mine aren't so now if I train the network, only parameters of my classifier will be updated not the features layer.
model.classifier = Classifier(1024, 102, 1000)

class Arguments():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)