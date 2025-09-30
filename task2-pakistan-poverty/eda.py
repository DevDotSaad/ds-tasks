
import matplotlib.pyplot as plt

def plot_series(df, xcol, ycol, title, out_path):
    fig = plt.figure(figsize=(8,4))
    plt.plot(df[xcol], df[ycol], marker='o')
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
