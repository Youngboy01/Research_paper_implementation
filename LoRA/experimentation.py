import torch
import torch.nn as nn
from baseModel import MLP, LoRAMLP
from dataloader import MNISTData
from training import Trainer
from utils import count_trainable_parameters
import time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = MNISTData(batch_size=64)
    train_dl = data.get_train_dl()
    test_dl = data.get_test_dl()

    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 5

    print("EXP 1: Standard MLP")

    base_model = MLP(input_dim=784, hidden_dim=512, output_dim=10)
    base_params = count_trainable_parameters(base_model)
    print(f"Trainable parameters in MLP: {base_params}")

    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
    base_trainer = Trainer(
        base_model, train_dl, test_dl, loss_fn, base_optimizer, device
    )

    start_time = time.time()
    base_trainer.train(num_epochs)
    base_trainer.evaluate()
    end_time = time.time()
    print(f"Training time for MLP: {end_time - start_time:.2f} seconds\n")

    print("EXP 2: LoRA MLP")

    lora_model = LoRAMLP(input_dim=784, hidden_dim=512, output_dim=10)
    lora_params = count_trainable_parameters(lora_model)
    print(f"Trainable parameters in LoRA MLP: {lora_params}")

    lora_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, lora_model.parameters()), lr=0.001
    )
    lora_trainer = Trainer(
        lora_model, train_dl, test_dl, loss_fn, lora_optimizer, device
    )

    start_time = time.time()
    lora_trainer.train(num_epochs)
    lora_trainer.evaluate()
    end_time = time.time()
    print(f"Training time for LoRA MLP: {end_time - start_time:.2f} seconds\n")

    print("Summary:")
    print(f"MLP parameters: {base_params} | LoRA MLP parameters: {lora_params}")
    print(
        f"parameter reduction in lora: {100 * (base_params - lora_params) / base_params:.2f}%"
    )
    print(
        f"MLP training time: {end_time - start_time:.2f} seconds | LoRA MLP training time: {end_time - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
