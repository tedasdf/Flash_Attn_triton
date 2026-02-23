import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
SEQ_LEN = 16
BLOCK_SIZE = 2 
D_MODEL = 20

num_q_blocks = math.ceil(SEQ_LEN / BLOCK_SIZE)
num_kv_blocks = math.ceil(SEQ_LEN / BLOCK_SIZE)

# Initialize data
Q_global = np.random.randn(SEQ_LEN, D_MODEL)
K_global = np.random.randn(SEQ_LEN, D_MODEL)
V_global = np.random.randn(SEQ_LEN, D_MODEL)
O_global = np.zeros((SEQ_LEN, D_MODEL)) # This will be filled as we go

# Stats for the online softmax (one per row in the current Q_block)
m_i = np.full(BLOCK_SIZE, -np.inf)
l_i = np.zeros(BLOCK_SIZE)

# State
q_idx = 0
kv_idx = 0

def draw_iteration():
    plt.clf()
    # Layout
    ax_q = plt.subplot2grid((3, 4), (0, 0))
    ax_k = plt.subplot2grid((3, 4), (1, 0))
    ax_v = plt.subplot2grid((3, 4), (2, 0))
    ax_local_o = plt.subplot2grid((3, 4), (1, 1), colspan=2)
    ax_global_o = plt.subplot2grid((3, 4), (0, 3), rowspan=3)

    q_start = q_idx * BLOCK_SIZE
    q_end = min(q_start + BLOCK_SIZE, SEQ_LEN)
    kv_start = kv_idx * BLOCK_SIZE
    kv_end = min(kv_start + BLOCK_SIZE, SEQ_LEN)

    # 1. Inputs (HBM)
    def plot_matrix(ax, data, title, r_range, c_range, color):
        ax.imshow(data, cmap='Greys', alpha=0.2)
        rect = plt.Rectangle((c_range[0]-0.5, r_range[0]-0.5), 
                             c_range[1]-c_range[0], r_range[1]-r_range[0], 
                             edgecolor=color, fill=True, facecolor=color, alpha=0.4)
        ax.add_patch(rect)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plot_matrix(ax_q, Q_global, "Q Load", (q_start, q_end), (0, D_MODEL), 'red')
    plot_matrix(ax_k, K_global, "K Load", (kv_start, kv_end), (0, D_MODEL), 'green')
    plot_matrix(ax_v, V_global, "V Load", (kv_start, kv_end), (0, D_MODEL), 'orange')

    # 2. Local Accumulator (SRAM)
    # We "simulate" values being updated
    local_o = np.random.uniform(0.1, 1.0, size=(q_end-q_start, D_MODEL))
    ax_local_o.imshow(local_o, cmap='plasma')
    ax_local_o.set_title(f"SRAM: O_block (Step {kv_idx+1}/{num_kv_blocks})", fontweight='bold')
    
    # Add text for the online softmax stats
    info_text = f"Online Softmax Stats (m_i, l_i):\n"
    info_text += f"m_i: {np.random.uniform(2, 5, 3).round(2)}...\n"
    info_text += f"l_i: {np.random.uniform(10, 50, 3).round(2)}..."
    ax_local_o.text(0, -1, info_text, fontsize=8, color='blue', transform=ax_local_o.transData)

    # 3. Global Output (HBM)
    ax_global_o.imshow(O_global, cmap='Blues')
    out_rect = plt.Rectangle((-0.5, q_start-0.5), D_MODEL, q_end-q_start, 
                              edgecolor='blue', fill=False, lw=2)
    ax_global_o.add_patch(out_rect)
    ax_global_o.set_title("Global O (Final HBM)", fontsize=10)

    plt.suptitle(f"Single SM: Processing Q-Rows {q_start}-{q_end}\nInner Loop: Processing K-Rows {kv_start}-{kv_end}", y=0.98)
    plt.tight_layout()
    plt.draw()

def on_click(event):
    global q_idx, kv_idx, O_global
    
    # Logic: Update the Global O matrix ONLY when the full KV sweep for a Q-block is done
    if kv_idx == num_kv_blocks - 1:
        q_start = q_idx * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, SEQ_LEN)
        # Assign values into O to show it's "finished"
        O_global[q_start:q_end, :] = np.random.uniform(0.5, 1.0, (q_end-q_start, D_MODEL))
        
    kv_idx += 1
    if kv_idx >= num_kv_blocks:
        kv_idx = 0
        q_idx += 1 
    
    if q_idx >= num_q_blocks:
        print("Simulation Finished. All O values written to HBM.")
        plt.close()
        return
    draw_iteration()

fig = plt.figure(figsize=(12, 8))
fig.canvas.mpl_connect('button_press_event', on_click)
draw_iteration()
plt.show()