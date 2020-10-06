from globals import *
from utils.utils import *
from utils.visualisation import show_evaluation

from model import SeqForecast

if __name__ == "__main__":
    # Load the dataset
    dataset, scaler = load_dataset(config.dataset_path, show_data=False)
    # Prepare the dataset for training/testing
    subsequences = extract_subsequences(dataset, lag=3)
    # Split the dataset into train/test set
    train_set, test_set = train_test_split(subsequences)
    # Create new instance of RNN
    net = SeqForecast(input_dim=train_set['X'].shape[-1],
                      hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers)

    # Train the network
    if config.mode == "train":
        train_loop(net, config.epochs, config.lr, config.wd,
                   train_set, test_set, debug=True)
    else:
        # Load pretrained weights
        net.load_state_dict(torch.load(
            config.pretrained_path, map_location=device))

    show_evaluation(net, dataset, scaler, debug=True)
    torch.save(net.state_dict(), 'car_sales.pt')
