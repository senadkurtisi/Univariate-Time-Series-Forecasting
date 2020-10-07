from globals import *
from utils.utils import *
from utils.visualisation import show_evaluation

from model import SeqForecast

if __name__ == "__main__":
    # Load the dataset
    dataset, scaler = load_dataset(config.dataset_path, show_data=True)
    # Prepare the dataset for training/testing
    subsequences = extract_subsequences(dataset, lag=3)
    # Split the dataset into train/test set
    train_set, test_set = train_test_split(subsequences)

    # Train the network
    if config.mode == "train":
        # Create new instance of RNN
        net = SeqForecast(input_dim=train_set['X'].shape[-1],
                          hidden_dim=config.hidden_dim.default,
                          num_layers=config.num_layers)
        train_loop(net, config.epochs, config.lr, config.wd,
                   train_set, test_set, debug=True)
    else:
        # Create new instance of RNN using default values
        net = SeqForecast(input_dim=train_set['X'].shape[-1],
                          hidden_dim=parser.get_default('hidden_dim'),
                          num_layers=parser.get_default('num_layers'))
        # Load pretrained weights
        net.load_state_dict(torch.load(
            config.pretrained_path, map_location=device))

    # Display the prediction next to the target output values
    show_evaluation(net, dataset, scaler, debug=True)
