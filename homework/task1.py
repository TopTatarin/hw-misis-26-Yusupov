import statistics

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    # pin_memory кладёт батчи в page-locked память, откуда копирование на GPU
    # можно делать асинхронно (non_blocking). num_workers готовит следующие батчи
    # в фоне, чтобы GPU не простаивал в ожидании данных
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    # CUDA-события — честный способ замерить время на GPU. Обычный time.time()
    # фиксирует лишь момент постановки ядра в очередь, а само вычисление идёт
    # асинхронно, поэтому такие тайминги получаются заниженными и недостоверными
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        # Шум сразу генерируем на GPU, чтобы не создавать тензор на CPU
        # и не гонять его лишний раз через шину PCIe
        noise = torch.randn(data.shape, device='cuda')
        # non_blocking в паре с pin_memory позволяет копировать данные
        # параллельно с вычислениями, а не блокировать поток на каждом батче
        data = data.to('cuda', non_blocking=True) + noise
        target = target.to('cuda', non_blocking=True)

        optimizer.zero_grad()

        fwd_start.record()
        output = model(data)
        loss = criterion(output, target)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()

        # Дожидаемся, пока GPU реально досчитает все поставленные ядра,
        # и только после этого снимаем тайминги
        torch.cuda.synchronize()
        forward_times.append(fwd_start.elapsed_time(fwd_end))
        backward_times.append(bwd_start.elapsed_time(bwd_end))

        # остаётся весь граф вычислений каждого батча → утечка и неминуемый OOM
        losses_history.append(loss.item())
        print(f"Batch {batch_idx} loss: {losses_history[-1]:.4f}")
        # torch.cuda.empty_cache() из цикла убран он принудительно синхронизирует поток, это заметно замедляет обучение, а от утечек всё равно не спасает

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()
