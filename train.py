import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import functional as SF
import tonic
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Headless plotting for cloud/server environments
plt.switch_backend('Agg')

# --- CONFIGURATION ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 5
DATA_PATH = '/data'
TIME_STEPS = 15  # Forces variable-length events into fixed time bins

def save_gif(data_tensor, label, pred, index, save_folder):
    """Generates a GIF of the event stream with prediction overlay."""
    # Combine polarities (On/Off events) for contrast
    frames = (data_tensor[:, 1, :, :] - data_tensor[:, 0, :, :]).cpu().numpy()
    
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap='seismic', vmin=-1, vmax=1)
    ax.axis('off')
    ax.set_title(f"True: {label} | Pred: {pred}")
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    filename = f"{save_folder}/sample_{index}_True_{label}_Pred_{pred}.gif"
    ani.save(filename, writer='pillow', fps=5)
    print(f"   Saved {filename}")
    plt.close(fig)

def check_accuracy(loader, net):
    total = 0
    correct = 0
    net.eval()
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device).permute(1, 0, 2, 3, 4).float()
            targets = targets.to(device)
            spk_rec, _ = net(data)
            acc = SF.accuracy_rate(spk_rec, targets)
            correct += acc * BATCH_SIZE
            total += BATCH_SIZE
    return correct / total

# --- DATA PIPELINE ---
print(f"✅ Using device: {device}")
sensor_size = tonic.datasets.NMNIST.sensor_size
transform = tonic.transforms.Compose([
    tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=TIME_STEPS),
    torch.from_numpy,
])

# Load N-MNIST 
train_set = tonic.datasets.NMNIST(save_to=DATA_PATH, train=True, transform=transform)
test_set = tonic.datasets.NMNIST(save_to=DATA_PATH, train=False, transform=transform)

# Disk Caching for speed
train_set = tonic.DiskCachedDataset(train_set, cache_path=os.path.join(DATA_PATH, 'cache_train'))
test_set = tonic.DiskCachedDataset(test_set, cache_path=os.path.join(DATA_PATH, 'cache_test'))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# --- NETWORK ARCHITECTURE (LIF SNN) ---
num_inputs = 2 * 34 * 34
num_hidden = 1000
num_outputs = 10
beta = 0.9  # Decay rate for Leaky Integrate-and-Fire neurons

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []

        # x shape: [Time, Batch, Channels, Height, Width]
        for step in range(x.size(0)):
            cur_frame = x[step].view(BATCH_SIZE, -1)
            cur1, mem1 = self.lif1(self.fc1(cur_frame), mem1)
            spk2, mem2 = self.lif2(self.fc2(cur1), mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

# --- TRAINING LOOP ---
print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    net.train()
    total_loss = 0
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device).permute(1, 0, 2, 3, 4).float()
        targets = targets.to(device)

        spk_rec, _ = net(data)
        loss = loss_fn(spk_rec, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader)
    print(f"✅ Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")
    
    # Run test set evaluation
    test_acc = check_accuracy(test_loader, net)
    print(f"   🏆 Test Set Accuracy: {test_acc * 100:.2f}%")

# --- GENERATE OUTPUTS ---
print("\n🎥 Generating Visualizations...")
visual_path = os.path.join(DATA_PATH, "visuals")
os.makedirs(visual_path, exist_ok=True)

net.eval()
with torch.no_grad():
    data, targets = next(iter(test_loader))
    data = data.to(device).permute(1, 0, 2, 3, 4).float()
    spk_rec, _ = net(data)
    preds = spk_rec.sum(dim=0).argmax(dim=1)
    
    for i in range(10):
        save_gif(data[:, i], targets[i].item(), preds[i].item(), i, visual_path)

save_path = os.path.join(DATA_PATH, "snn_model.pth")
torch.save(net.state_dict(), save_path)
print(f"💾 Model saved to {save_path}")
