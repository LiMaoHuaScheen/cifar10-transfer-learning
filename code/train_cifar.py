"""
CIFAR-10 图像分类实验：从头训练 vs 迁移学习微调 (ResNet-18)
产出：测试准确率对比、学习曲线、混淆矩阵
考研AI方向简历项目 · 方案A第一步
"""
import os, time, argparse, copy, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_loaders(root, batch_size, workers):
    tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10(root=root, train=True, download=False, transform=tr)
    test_ds = datasets.CIFAR10(root=root, train=False, download=False, transform=te)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, train_ds.classes

def build_model(mode, num_classes=10):
    if mode == "scratch":
        return models.resnet18(weights=None, num_classes=num_classes), "从头训练(from scratch)"
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, "迁移学习(pretrained+finetune)"

def evaluate(model, loader, device):
    model.eval(); correct=0; total=0; preds=[]; labels=[]
    with torch.no_grad():
        for images, lb in loader:
            images, lb = images.to(device), lb.to(device)
            out = model(images); _, p = torch.max(out,1)
            correct += (p==lb).sum().item(); total += lb.size(0)
            preds.extend(p.cpu().numpy()); labels.extend(lb.cpu().numpy())
    return 100.0*correct/total, np.array(preds), np.array(labels)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); running=0.0; correct=0; total=0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); out = model(images); loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        running += loss.item()*images.size(0)
        _, p = torch.max(out,1); correct += (p==labels).sum().item(); total += labels.size(0)
    return running/total, 100.0*correct/total

def run_experiment(mode, args, train_loader, test_loader, device):
    model, desc = build_model(mode); model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = args.lr*2 if mode=="scratch" else args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"\n{'='*60}\n[{desc}] 训练 {args.epochs} epochs, lr={lr}, device={device}\n{'='*60}", flush=True)
    train_accs=[]; test_accs=[]; best=0.0; best_state=None
    for epoch in range(1, args.epochs+1):
        t0=time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        tea, _, _ = evaluate(model, test_loader, device)
        scheduler.step()
        train_accs.append(ta); test_accs.append(tea)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={tl:.4f} | train_acc={ta:.2f}% | test_acc={tea:.2f}% | {time.time()-t0:.1f}s", flush=True)
        if tea > best:
            best = tea; best_state = copy.deepcopy(model.state_dict())
    if best_state is not None: model.load_state_dict(best_state)
    final_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"--> [{desc}] 最佳测试准确率: {best:.2f}%", flush=True)
    return {"mode":mode,"desc":desc,"train_accs":train_accs,"test_accs":test_accs,"best_test_acc":best,"preds":preds,"labels":labels}

def main():
    args = parse_args(); set_seed(args.seed)
    device = torch.device(args.device); print(f"Device: {device}")
    root = args.data_root
    train_loader, test_loader, classes = build_loaders(root, args.batch_size, args.workers)
    print(f"Classes: {classes}")
    results = {}
    results["scratch"] = run_experiment("scratch", args, train_loader, test_loader, device)
    results["finetune"] = run_experiment("finetune", args, train_loader, test_loader, device)
    out = {"scratch_best_acc": results["scratch"]["best_test_acc"],
           "finetune_best_acc": results["finetune"]["best_test_acc"],
           "scratch_train_accs": results["scratch"]["train_accs"],
           "scratch_test_accs": results["scratch"]["test_accs"],
           "finetune_train_accs": results["finetune"]["train_accs"],
           "finetune_test_accs": results["finetune"]["test_accs"]}
    os.makedirs("output", exist_ok=True)
    with open("output/result_summary.json","w") as f: json.dump(out, f, indent=2)
    print("\n"+"="*60)
    print(f"从头训练最佳准确率 : {results['scratch']['best_test_acc']:.2f}%")
    print(f"迁移学习最佳准确率 : {results['finetune']['best_test_acc']:.2f}%")
    print(f"提升幅度           : {results['finetune']['best_test_acc']-results['scratch']['best_test_acc']:+.2f} 个百分点")
    print("结果已保存到 output/result_summary.json")
    print("="*60)

if __name__ == "__main__":
    main()
