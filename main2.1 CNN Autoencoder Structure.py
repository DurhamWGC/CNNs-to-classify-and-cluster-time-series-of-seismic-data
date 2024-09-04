# main2.1 CNN autoencoder structure plot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_block(ax, label, position, size=(2, 2), color='lightblue', text_color='black', fontsize=10):
    """Draws a block representing a layer"""
    ax.add_patch(patches.Rectangle(position, size[0], size[1], edgecolor='black', facecolor=color))
    ax.text(position[0] + size[0]/2, position[1] + size[1]/2, label, 
            verticalalignment='center', horizontalalignment='center', fontsize=fontsize, color=text_color, weight='bold')

def plot_resnet_autoencoder_structure():
    fig, ax = plt.subplots(figsize=(24, 12))  # Higher resolution
    
    center_y = 6  # Central y position for alignment

    block_widths = [2, 2, 2, 2, 2, 1.5, 1.5, 2, 2, 2, 2]  # Block sizes
    block_positions = []
    current_x = 0

    # Calculate positions for each layer
    for width in block_widths:
        block_positions.append(current_x)
        current_x += width + 0.5  # Add spacing for arrows
    
    # Input layer
    draw_block(ax, 'Input\n(128x1)', (block_positions[0], center_y - 2), size=(block_widths[0], 4), color='lightgreen')
    
    # Encoder layers
    draw_block(ax, 'Conv1D\n(128x32)', (block_positions[1], center_y - 2), size=(block_widths[1], 4), color='lightblue', text_color='black', fontsize=9)
    draw_block(ax, 'MaxPooling\n(64x32)', (block_positions[2], center_y - 1.5), size=(block_widths[2], 3), color='lightcoral', text_color='black', fontsize=9)

    # Residual Block 1
    draw_block(ax, 'ResBlock\n(64x32)', (block_positions[3], center_y - 2), size=(block_widths[3], 4), color='lightblue', text_color='black', fontsize=9)
    draw_block(ax, 'MaxPooling\n(32x32)', (block_positions[4], center_y - 1.5), size=(block_widths[4], 3), color='lightcoral', text_color='black', fontsize=9)

    # Residual Block 2
    draw_block(ax, 'ResBlock\n(32x64)', (block_positions[5], center_y - 2), size=(block_widths[5], 4), color='lightblue', text_color='black', fontsize=9)
    draw_block(ax, 'MaxPooling\n(16x64)', (block_positions[6], center_y - 1.5), size=(block_widths[6], 3), color='lightcoral', text_color='black', fontsize=9)

    # Flatten and Dense layers
    draw_block(ax, 'Flatten\n(1024)', (block_positions[7], center_y - 1), size=(block_widths[7], 2), color='lightpink', text_color='black', fontsize=9)
    draw_block(ax, 'Dense\n(64)', (block_positions[8], center_y - 1), size=(block_widths[8], 2), color='lightpink', text_color='black', fontsize=9)

    # Decoder layers
    draw_block(ax, 'Reshape\n(16x64)', (block_positions[9], center_y - 1), size=(block_widths[9], 2), color='lightblue', text_color='black', fontsize=9)
    draw_block(ax, 'Conv1D\n(16x64)', (block_positions[10], center_y - 2), size=(block_widths[10], 4), color='lightblue', text_color='black', fontsize=9)
    draw_block(ax, 'UpSampling\n(32x64)', (block_positions[9], center_y - 1.5), size=(block_widths[9], 3), color='lightcoral', text_color='black', fontsize=9)

    draw_block(ax, 'Conv1D\n(32x32)', (block_positions[10], center_y - 2), size=(block_widths[10], 4), color='lightblue', text_color='black', fontsize=9)
    draw_block(ax, 'UpSampling\n(64x32)', (block_positions[10], center_y - 1.5), size=(block_widths[10], 3), color='lightcoral', text_color='black', fontsize=9)
    
    # Output layer
    draw_block(ax, 'Output\n(128x1)', (block_positions[10], center_y - 2), size=(block_widths[10], 4), color='lightgray', text_color='black', fontsize=9)
    
    # Add arrows between layers
    for i in range(len(block_positions) - 1):
        start_x = block_positions[i] + block_widths[i]  # Start at the right of each block
        ax.arrow(start_x, center_y, 0.5, 0, head_width=0.15, head_length=0.15, fc='grey', ec='grey')
    
    # Set plot limits and remove axes
    ax.set_xlim(-1, current_x + 1)
    ax.set_ylim(3, 9)
    ax.axis('off')
    
    # Add title
    plt.figtext(0.5, 0.02, 'CNN Autoencoder Structure', ha='center', fontsize=16, weight='bold')
    
    plt.show()

plot_resnet_autoencoder_structure()
