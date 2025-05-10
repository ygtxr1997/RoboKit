import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def dataloader_speed_test(dataset: Dataset, batch_size=64, num_workers=8, num_batches=1000):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # tqdm 参数说明：
    # total: 总迭代次数，这里等于 dataset_size // batch_size（向上取整）
    total_iters = (len(dataset) + batch_size - 1) // batch_size
    desc = f"workers={num_workers}, batch={batch_size}"

    start_time = time.perf_counter()
    total_batches = 0

    # 在 loader 上包一层 tqdm
    for batch_data in tqdm(loader, total=total_iters, desc=desc):
        total_batches += 1
        # 如果有 GPU 拷贝需求可以取消以下注释
        # _ = images.cuda(non_blocking=True)
        if total_batches >= num_batches:
            break

    elapsed = time.perf_counter() - start_time
    print(f"{desc} ⇒ Samples={total_batches}, Time={elapsed:.3f}s, "
          f"Throughput={total_batches/elapsed:.1f} batches/s")