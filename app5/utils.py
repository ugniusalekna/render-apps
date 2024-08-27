import torch
import json

from model import DoodlesClassifier


class Args:
    def __init__(self):
        self.image_size = (64, 64)
        self.channels_in = 1
        self.hidden_layers = [16, 32, 64, 128]
        self.classes = [
            "cat",
            "dog",
            "bicycle",
            "airplane",
            "tree",
            "house",
            "flower",
            "fish",
            "car",
            "apple"
        ]
        self.num_classes = len(self.classes)
        
    @staticmethod
    def load(load_path):
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        args = Args()
        args.image_size = tuple(data.get('image_size', args.image_size))
        args.channels_in = data.get('channels_in', args.channels_in)
        args.hidden_layers = data.get('hidden_layers', args.hidden_layers)
        args.classes = data.get('classes', args.classes)
        args.num_classes = len(args.classes)
        
        return args
    
    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
            
def load_args_and_model(load_path, device):
    args = Args.load(load_path + "/model_args.json")
    state_dict = torch.load(load_path + "/model_state_dict.pth", map_location=device, weights_only=True)

    model = DoodlesClassifier(args.image_size, args.channels_in, args.hidden_layers, args.num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return args, model


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    