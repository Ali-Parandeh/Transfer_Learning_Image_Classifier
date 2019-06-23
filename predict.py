# PROGRAMMER: Alireza Parandeh
# DATE CREATED: 22.06.2019
# REVISED DATE:
# PURPOSE: Predict flower name from an image along with the probability of that name.
#          That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

from args import get_prediction_input_args
import torch
import json
import numpy as np
from PIL import Image
from torchvision import models
from classifier import Classifier



# Writing a function that loads a checkpoint and rebuilds the model
def checkpoint_loader(filepath, model_name):
    ''' Constructs the fully trained model from a saved checkpoint.
    '''
    checkpoint = torch.load(
        filepath, map_location=lambda storage, loc: storage)

    if model_name:
        model = checkpoint["model"]
    else:
        model = models.densenet121(pretrained=True)
        model.classifier = Classifier(checkpoint['input_size'],
                                      checkpoint['output_size'],
                                      checkpoint['hidden_layer'],
                                      checkpoint['dropout'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)

    # Resizes image so shortest side is 256px while keeping aspect ratio (Image.thumbnail)
    pil_image.thumbnail([pil_image.size[0], 256])

    # Crop out the center of 224x224 portion of the image
    w, h = pil_image.size
    i = int(round((w - 224)/2))  # x coordinate
    j = int(round((h - 224)/2))  # y coordinate
    # x-coord, y-coord, x+crop_size coord, y=crop_size coord)
    pil_image = pil_image.crop([i, j, i+224, j+224])

    # Normalise color channels to range [0, 1] - Use np.array(pil_image)
    np_image = np.array(pil_image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose array to what Pytorch expects using ndarray.tranpose() from (W, H, Color) to (color, h, w)
    np_image = np_image.transpose((2, 0, 1))

    # Convert to numpy array to torch
    tensor_image = torch.from_numpy(np_image)

    return tensor_image


def load_labels(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def predict(image_path, model, gpu, json_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    image = process_image(image_path)

    # The model needs to know the batch size as well so tensor shape is (batch_no, c, h, w)
    image.unsqueeze_(0)

    # Casting the inputs to torch.FloatTensor as they're the default tensor type for weights and biases
    image = image.float()

    # Send the image tensor and the model to CPU or GPU for prediction depending user input and GPU availability
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    image.to(device)

    # Feedforward the image through the model to get probabilities and classes
    model.eval()
    ps = torch.exp(model.forward(image))
    topk_probs, topk_classes = ps.topk(topk, dim=1)

    # Converting tensors to standard python data structures
    probs = topk_probs.detach().numpy()[0]
    classes = topk_classes.numpy()[0]

    # Converting from class indices to class labels
    v = model.class_to_idx.values()
    k = model.class_to_idx.keys()
    idx_to_classes = dict([[v, k] for k, v in model.class_to_idx.items()])
    classes = [idx_to_classes[idx] for idx in classes]

    # Translate the image class indices to class names and save in a dictionary
    cat_to_name = load_labels(json_path)
    top_five_classes = []
    for i in range(len(classes)):
        top_five_classes.append(cat_to_name[classes[i]])

    return top_five_classes, probs


def main():
    pi = get_prediction_input_args()

    model = checkpoint_loader(pi.checkpoint, pi.arch)
    prediction, probs = predict(
        pi.input, model, pi.gpu, pi.category_names, pi.top_k)

    print("The most likely class is {} with associated probability {}.".format(
        prediction[0], probs[0]))
    print("The other top {} classes include the following: ".format(pi.top_k - 1))
    for i in range(len(prediction)-1):
        print("Class Name: {}    Probability: {}". format(
            prediction[i+1], probs[i+1]))


if __name__ == "__main__":
    main()
