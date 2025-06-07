from training.run_experiments import run_experiments

from src.core.config import get_config


if __name__ == "__main__":

    print('Running training experiments ...')

    config = get_config()

    # Data
    train_file = config['paths']['training_data']
    val_file = config['paths']['validation_data']
    test_file = config['paths']['testing_data']

    # Model parameters
    input_size = config['model']['input_size']
    hidden_dim = config['model']['hidden_dim']
    num_layer = config['model']['num_layers']

    # Training hyperparameters
    window_sizes = config['training']['window_sizes']
    step_sizes = config['training']['step_sizes']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    # Log to Tensorboard
    log_dir = config['paths']['log_dir']
    
    # Model save path
    model_save_path = config['paths']['model_save_path']

    # Threshold percentile
    threshold_percentile = config['inference']['threshold_percentile']

    run_experiments(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        window_sizes=window_sizes,
        step_sizes=step_sizes,
        input_size=input_size,
        hidden_dim=hidden_dim,
        num_layers=num_layer,
        epochs=epochs,
        batch_size=batch_size,
        log_dir=log_dir,
        model_save_path=model_save_path,
        threshold_percentile=threshold_percentile
    )

    print('Done')