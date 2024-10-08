# main2.1 CNN autoencoder structure plot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_block(ax, label, position, size=(2, 2), color='lightblue', text_color='black', fontsize=9):
    """Draws a block representing a layer"""
    ax.add_patch(patches.Rectangle(position, size[0], size[1], edgecolor='black', facecolor=color))
    ax.text(position[0] + size[0]/2, position[1] + size[1]/2, label, 
            verticalalignment='center', horizontalalignment='center', fontsize=fontsize, color=text_color, weight='bold')

def plot_resnet_autoencoder_structure():
    # Increase the width to accommodate more blocks
    fig, ax = plt.subplots(figsize=(50, 20))  # Increase the width to 50

    center_y = 6  # Central y position for alignment

    # Define layer block sizes (extended to include more layers)
    block_widths = [1, 1, 1,  1,  1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1]
    block_positions = []
    current_x = 0

    # Calculate positions for each block
    for width in block_widths:
        block_positions.append(current_x)
        current_x += width + 0.3  # Add spacing for arrows
    
    # Input layer (larger fontsize for main layers)
    draw_block(ax, 'Input\n(128x1)', (block_positions[0], center_y - 1.5), size=(block_widths[0], 3), color='lightgreen', fontsize=24)
    
    # Encoder layers
    draw_block(ax, 'Conv1D\n(128x32)', (block_positions[1], center_y - 1.5), size=(block_widths[1], 3), color='lightblue', text_color='black', fontsize=20)
    draw_block(ax, 'MaxPooling\n(64x32)', (block_positions[2], center_y - 1.25), size=(block_widths[2], 2.5), color='lightcoral', text_color='black', fontsize=20)

    # Residual Block 1
    draw_block(ax, 'ResBlock\n(64x32)', (block_positions[3], center_y - 1.5), size=(block_widths[3], 3), color='lightblue', text_color='black', fontsize=20)
    draw_block(ax, 'MaxPooling\n(32x32)', (block_positions[4], center_y - 1.25), size=(block_widths[4], 2.5), color='lightcoral', text_color='black', fontsize=20)

    # Residual Block 2
    draw_block(ax, 'ResBlock\n(32x64)', (block_positions[5], center_y - 1.5), size=(block_widths[5], 3), color='lightblue', text_color='black', fontsize=20)
    draw_block(ax, 'MaxPooling\n(16x64)', (block_positions[6], center_y - 1.25), size=(block_widths[6], 2.5), color='lightcoral', text_color='black', fontsize=20)

    # End of encoder
    draw_block(ax, 'Flatten\n(1024)', (block_positions[7], center_y - 1), size=(block_widths[7], 2), color='lightpink', text_color='black', fontsize=20)
    draw_block(ax, 'Dense\n(64)', (block_positions[8], center_y - 1), size=(block_widths[8], 2), color='lightpink', text_color='black', fontsize=20)

    # Decoder layers (smaller fontsize for secondary layers)
    draw_block(ax, 'Reshape\n(16x64)', (block_positions[9], center_y - 1), size=(block_widths[9], 2), color='lightblue', text_color='black', fontsize=20)
    draw_block(ax, 'Conv1D\n(16x64)', (block_positions[10], center_y - 1.5), size=(block_widths[10], 3), color='lightblue', text_color='black', fontsize=20)
    draw_block(ax, 'UpSampling\n(32x64)', (block_positions[11], center_y - 1.5), size=(block_widths[11], 3), color='lightcoral', text_color='black', fontsize=20)

    draw_block(ax, 'Conv1D\n(32x32)', (block_positions[12], center_y - 1.5), size=(block_widths[12], 3), color='lightblue', text_color='black', fontsize=20)
    draw_block(ax, 'UpSampling\n(64x32)', (block_positions[13], center_y - 1.5), size=(block_widths[13], 3), color='lightcoral', text_color='black', fontsize=20)
    
    # Output layer (larger fontsize for main layers)
    draw_block(ax, 'Output\n(128x1)', (block_positions[14], center_y - 1.5), size=(block_widths[14], 3), color='lightgray', text_color='black', fontsize=24)
    
    # Add arrows between blocks
    for i in range(len(block_positions) - 1):
        start_x = block_positions[i] + block_widths[i]
        ax.arrow(start_x, center_y, 0.3, 0, head_width=0.1, head_length=0.1, fc='grey', ec='grey')
    
    # Set plot limits and remove axes
    ax.set_xlim(-1, current_x + 1)
    ax.set_ylim(4, 8)
    ax.axis('off')
    
    # Add title at the bottom
    plt.figtext(0.5, 0.1, 'CNN Autoencoder Structure', ha='center', fontsize=38, weight='bold')
    
    plt.show()

plot_resnet_autoencoder_structure()
