#!/usr/bin/env python3
"""
CUDA卷积中Halo Cells概念的可视化演示
展示不同策略处理halo cells的方式
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ConvolutionVisualizer:
    def __init__(self, input_size=(8, 8), tile_size=4, filter_radius=1):
        self.input_size = input_size
        self.tile_size = tile_size
        self.filter_radius = filter_radius
        self.filter_size = 2 * filter_radius + 1
        self.block_size = tile_size + 2 * filter_radius
        
        # 创建输入数据
        self.input_data = np.arange(input_size[0] * input_size[1]).reshape(input_size)
        
        # 创建简单的平均滤波器
        self.filter = np.ones((self.filter_size, self.filter_size)) / (self.filter_size ** 2)
        
    def visualize_tiles_and_halos(self):
        """可视化tiles和halo cells"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('CUDA Convolution: Tiles and Halo Cells Visualization', fontsize=16)
        
        # 计算网格尺寸
        grid_x = (self.input_size[1] + self.tile_size - 1) // self.tile_size
        grid_y = (self.input_size[0] + self.tile_size - 1) // self.tile_size
        
        for idx, (tile_y, tile_x) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            if tile_y >= grid_y or tile_x >= grid_x:
                axes[tile_y, tile_x].axis('off')
                continue
                
            ax = axes[tile_y, tile_x]
            self.visualize_single_tile(ax, tile_x, tile_y)
            ax.set_title(f'Tile [{tile_x}, {tile_y}]')
        
        plt.tight_layout()
        return fig
    
    def visualize_single_tile(self, ax, tile_x, tile_y):
        """可视化单个tile及其halo"""
        # 创建颜色映射
        data_vis = np.zeros(self.input_size)
        colors = np.zeros((*self.input_size, 3))
        
        # 计算tile的范围
        tile_start_x = tile_x * self.tile_size
        tile_start_y = tile_y * self.tile_size
        tile_end_x = min(tile_start_x + self.tile_size - 1, self.input_size[1] - 1)
        tile_end_y = min(tile_start_y + self.tile_size - 1, self.input_size[0] - 1)
        
        # 计算包含halo的范围
        halo_start_x = max(0, tile_start_x - self.filter_radius)
        halo_start_y = max(0, tile_start_y - self.filter_radius)
        halo_end_x = min(self.input_size[1] - 1, tile_end_x + self.filter_radius)
        halo_end_y = min(self.input_size[0] - 1, tile_end_y + self.filter_radius)
        
        # 设置颜色
        # 内部元素 - 绿色
        for y in range(tile_start_y, tile_end_y + 1):
            for x in range(tile_start_x, tile_end_x + 1):
                colors[y, x] = [0.2, 0.8, 0.2]  # 绿色
                data_vis[y, x] = self.input_data[y, x]
        
        # Halo cells - 黄色
        for y in range(halo_start_y, halo_end_y + 1):
            for x in range(halo_start_x, halo_end_x + 1):
                if not (tile_start_y <= y <= tile_end_y and tile_start_x <= x <= tile_end_x):
                    colors[y, x] = [0.9, 0.9, 0.2]  # 黄色
                    data_vis[y, x] = self.input_data[y, x]
        
        # 显示数据
        im = ax.imshow(colors, interpolation='nearest', aspect='equal')
        
        # 添加数值标签
        for y in range(self.input_size[0]):
            for x in range(self.input_size[1]):
                if halo_start_y <= y <= halo_end_y and halo_start_x <= x <= halo_end_x:
                    text_color = 'black' if (tile_start_y <= y <= tile_end_y and 
                                            tile_start_x <= x <= tile_end_x) else 'black'
                    ax.text(x, y, f'{int(self.input_data[y, x])}', 
                           ha='center', va='center', color=text_color, fontsize=10)
        
        # 添加边框
        # 内部tile边框 - 粗绿线
        rect_interior = Rectangle((tile_start_x - 0.5, tile_start_y - 0.5),
                                 tile_end_x - tile_start_x + 1,
                                 tile_end_y - tile_start_y + 1,
                                 linewidth=3, edgecolor='green', facecolor='none')
        ax.add_patch(rect_interior)
        
        # Halo边框 - 虚线黄色
        rect_halo = Rectangle((halo_start_x - 0.5, halo_start_y - 0.5),
                             halo_end_x - halo_start_x + 1,
                             halo_end_y - halo_start_y + 1,
                             linewidth=2, edgecolor='orange', facecolor='none',
                             linestyle='--')
        ax.add_patch(rect_halo)
        
        # Ghost cells标记（如果有的话）
        ghost_start_x = tile_start_x - self.filter_radius
        ghost_start_y = tile_start_y - self.filter_radius
        ghost_end_x = tile_end_x + self.filter_radius
        ghost_end_y = tile_end_y + self.filter_radius
        
        for y in range(ghost_start_y, ghost_end_y + 1):
            for x in range(ghost_start_x, ghost_end_x + 1):
                if x < 0 or x >= self.input_size[1] or y < 0 or y >= self.input_size[0]:
                    # 标记ghost cell位置
                    ghost_x = max(0, min(x, self.input_size[1] - 1))
                    ghost_y = max(0, min(y, self.input_size[0] - 1))
                    
                    if x < 0:
                        ax.text(-0.7, ghost_y, 'G', ha='center', va='center', 
                               color='red', fontsize=8, weight='bold')
                    elif x >= self.input_size[1]:
                        ax.text(self.input_size[1] - 0.3, ghost_y, 'G', 
                               ha='center', va='center', color='red', fontsize=8, weight='bold')
                    elif y < 0:
                        ax.text(ghost_x, -0.7, 'G', ha='center', va='center', 
                               color='red', fontsize=8, weight='bold')
                    elif y >= self.input_size[0]:
                        ax.text(ghost_x, self.input_size[0] - 0.3, 'G', 
                               ha='center', va='center', color='red', fontsize=8, weight='bold')
        
        ax.set_xlim(-1, self.input_size[1])
        ax.set_ylim(self.input_size[0], -1)
        ax.set_xticks(range(self.input_size[1]))
        ax.set_yticks(range(self.input_size[0]))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
    def analyze_memory_access(self):
        """分析内存访问模式"""
        # 创建访问计数矩阵
        access_count = np.zeros(self.input_size)
        
        # 计算每个tile访问的元素
        grid_x = (self.input_size[1] + self.tile_size - 1) // self.tile_size
        grid_y = (self.input_size[0] + self.tile_size - 1) // self.tile_size
        
        for tile_y in range(grid_y):
            for tile_x in range(grid_x):
                # 计算halo范围
                halo_start_x = max(0, tile_x * self.tile_size - self.filter_radius)
                halo_start_y = max(0, tile_y * self.tile_size - self.filter_radius)
                halo_end_x = min(self.input_size[1] - 1, 
                               (tile_x + 1) * self.tile_size - 1 + self.filter_radius)
                halo_end_y = min(self.input_size[0] - 1, 
                               (tile_y + 1) * self.tile_size - 1 + self.filter_radius)
                
                # 更新访问计数
                for y in range(halo_start_y, halo_end_y + 1):
                    for x in range(halo_start_x, halo_end_x + 1):
                        access_count[y, x] += 1
        
        # 可视化访问模式
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Memory Access Pattern Analysis', fontsize=16)
        
        # 热力图
        sns.heatmap(access_count, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': 'Access Count'})
        ax1.set_title('Access Count Heatmap')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        
        # 统计信息
        unique_counts = np.unique(access_count)
        access_stats = [(count, np.sum(access_count == count)) 
                       for count in unique_counts]
        
        ax2.bar([str(int(s[0])) for s in access_stats], 
               [s[1] for s in access_stats])
        ax2.set_title('Access Count Distribution')
        ax2.set_xlabel('Number of Accesses')
        ax2.set_ylabel('Number of Elements')
        
        # 添加统计文本
        total_accesses = np.sum(access_count)
        unique_elements = np.prod(self.input_size)
        redundancy = total_accesses / unique_elements
        
        stats_text = f"Total Accesses: {int(total_accesses)}\n"
        stats_text += f"Unique Elements: {unique_elements}\n"
        stats_text += f"Redundancy Factor: {redundancy:.2f}x\n"
        stats_text += f"Average Accesses per Element: {np.mean(access_count):.2f}"
        
        ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def compare_strategies(self):
        """比较不同的halo处理策略"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Comparison of Halo Handling Strategies', fontsize=16)
        
        # 策略1: 所有数据在共享内存
        ax1 = axes[0, 0]
        self.visualize_strategy1(ax1)
        ax1.set_title('Strategy 1: All in Shared Memory')
        
        # 策略2: 只有内部元素在共享内存
        ax2 = axes[0, 1]
        self.visualize_strategy2(ax2)
        ax2.set_title('Strategy 2: Cache for Halos')
        
        # 内存使用对比
        ax3 = axes[1, 0]
        self.compare_memory_usage(ax3)
        ax3.set_title('Shared Memory Usage Comparison')
        
        # 性能分析
        ax4 = axes[1, 1]
        self.performance_analysis(ax4)
        ax4.set_title('Performance Analysis')
        
        plt.tight_layout()
        return fig
    
    def visualize_strategy1(self, ax):
        """可视化策略1：所有数据在共享内存"""
        block_data = np.zeros((self.block_size, self.block_size))
        colors = np.zeros((self.block_size, self.block_size, 3))
        
        # 内部元素 - 绿色
        for y in range(self.filter_radius, self.filter_radius + self.tile_size):
            for x in range(self.filter_radius, self.filter_radius + self.tile_size):
                colors[y, x] = [0.2, 0.8, 0.2]
        
        # Halo - 黄色
        for y in range(self.block_size):
            for x in range(self.block_size):
                if not (self.filter_radius <= y < self.filter_radius + self.tile_size and
                       self.filter_radius <= x < self.filter_radius + self.tile_size):
                    colors[y, x] = [0.9, 0.9, 0.2]
        
        ax.imshow(colors, interpolation='nearest')
        
        # 添加标签
        for y in range(self.block_size):
            for x in range(self.block_size):
                if self.filter_radius <= y < self.filter_radius + self.tile_size and \
                   self.filter_radius <= x < self.filter_radius + self.tile_size:
                    ax.text(x, y, 'I', ha='center', va='center', fontsize=12, weight='bold')
                else:
                    ax.text(x, y, 'H', ha='center', va='center', fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, -0.1, f'Shared Memory: {self.block_size}×{self.block_size} = {self.block_size**2} elements',
               transform=ax.transAxes, ha='center')
    
    def visualize_strategy2(self, ax):
        """可视化策略2：只有内部元素在共享内存"""
        tile_data = np.zeros((self.tile_size, self.tile_size))
        colors = np.ones((self.tile_size, self.tile_size, 3)) * [0.2, 0.8, 0.2]
        
        ax.imshow(colors, interpolation='nearest')
        
        for y in range(self.tile_size):
            for x in range(self.tile_size):
                ax.text(x, y, 'I', ha='center', va='center', fontsize=12, weight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, -0.1, f'Shared Memory: {self.tile_size}×{self.tile_size} = {self.tile_size**2} elements',
               transform=ax.transAxes, ha='center')
        ax.text(0.5, -0.2, 'Halos accessed via L1/L2 cache', 
               transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    def compare_memory_usage(self, ax):
        """比较内存使用"""
        strategies = ['Strategy 1\n(All in Shared)', 'Strategy 2\n(Cache Halos)']
        shared_mem = [self.block_size**2, self.tile_size**2]
        
        bars = ax.bar(strategies, shared_mem, color=['coral', 'lightgreen'])
        ax.set_ylabel('Shared Memory Elements')
        
        # 添加数值标签
        for bar, value in zip(bars, shared_mem):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value}', ha='center', va='bottom')
        
        # 计算节省百分比
        savings = (1 - self.tile_size**2 / self.block_size**2) * 100
        ax.text(0.5, 0.95, f'Strategy 2 saves {savings:.1f}% shared memory',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    def performance_analysis(self, ax):
        """性能分析"""
        categories = ['Shared Mem\nUsage', 'Programming\nComplexity', 
                     'Cache\nDependency', 'Control\nDivergence']
        
        strategy1 = [100, 80, 20, 60]  # 相对分数
        strategy2 = [40, 40, 80, 30]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, strategy1, width, label='Strategy 1', color='coral')
        ax.bar(x + width/2, strategy2, width, label='Strategy 2', color='lightgreen')
        
        ax.set_ylabel('Relative Score')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 120)
        
        # 添加说明
        ax.text(0.5, -0.3, 'Lower is better for all metrics',
               transform=ax.transAxes, ha='center', fontsize=10, style='italic')

def main():
    print("=" * 60)
    print("CUDA Convolution: Halo Cells Visualization")
    print("=" * 60)
    
    # 创建可视化器
    vis = ConvolutionVisualizer(input_size=(8, 8), tile_size=4, filter_radius=1)
    
    print("\nConfiguration:")
    print(f"- Input Size: {vis.input_size}")
    print(f"- Tile Size: {vis.tile_size}×{vis.tile_size}")
    print(f"- Filter Radius: {vis.filter_radius}")
    print(f"- Filter Size: {vis.filter_size}×{vis.filter_size}")
    print(f"- Block Size (with halo): {vis.block_size}×{vis.block_size}")
    
    # 生成可视化
    print("\nGenerating visualizations...")
    
    # 1. Tiles和Halos可视化
    fig1 = vis.visualize_tiles_and_halos()
    plt.savefig('/home/claude/tiles_and_halos.png', dpi=150, bbox_inches='tight')
    print("- Saved: tiles_and_halos.png")
    
    # 2. 内存访问模式分析
    fig2 = vis.analyze_memory_access()
    plt.savefig('/home/claude/memory_access_pattern.png', dpi=150, bbox_inches='tight')
    print("- Saved: memory_access_pattern.png")
    
    # 3. 策略比较
    fig3 = vis.compare_strategies()
    plt.savefig('/home/claude/strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("- Saved: strategy_comparison.png")
    
    # 计算统计信息
    print("\nStatistics:")
    grid_x = (vis.input_size[1] + vis.tile_size - 1) // vis.tile_size
    grid_y = (vis.input_size[0] + vis.tile_size - 1) // vis.tile_size
    total_tiles = grid_x * grid_y
    
    print(f"- Grid dimensions: {grid_x}×{grid_y} = {total_tiles} tiles")
    
    # 计算halo开销
    interior_elements = vis.tile_size ** 2
    total_elements = vis.block_size ** 2
    halo_elements = total_elements - interior_elements
    halo_percentage = (halo_elements / total_elements) * 100
    
    print(f"\nPer-tile analysis:")
    print(f"- Interior elements: {interior_elements}")
    print(f"- Halo elements: {halo_elements}")
    print(f"- Total elements needed: {total_elements}")
    print(f"- Halo overhead: {halo_percentage:.1f}%")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
