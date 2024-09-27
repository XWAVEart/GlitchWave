import os
import io
import cv2
import math
import noise
import random
import numpy as np
from PIL import Image
from collections import deque
from scipy.spatial import cKDTree
import colorsys  # For hue calculations


def load_image(file_path):
    """
    Loads an image from the given file path.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None
    try:
        image = Image.open(file_path)
        image = image.convert('RGB')  # Ensure image is in RGB mode
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def generate_output_filename(original_path, effect, settings):
    """
    Generates a filename for the output image based on the original filename, the effect applied, and settings.
    """
    base_name, extension = os.path.splitext(os.path.basename(original_path))
    directory = os.path.dirname(original_path)
    new_filename = f"{base_name}_{effect}_{settings}{extension}"
    return os.path.join(directory, new_filename)

def save_image(image, output_path):
    """
    Saves the image to the specified output path.
    """
    try:
        image.save(output_path)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")

def process_image():
    """
    Main function to process the image based on user input.
    """
    file_path = input("Enter the path of the image file: ")
    image = load_image(file_path)
    if image is None:
        return

    print("Choose an effect:")
    print("1. Pixel Sorting (Original)")
    print("2. Pixel Sorting (Corner-to-Corner)")
    print("3. Color Channel Manipulation")
    print("4. Data Moshing")
    print("5. Pixel Drift")
    print("6. Bit Manipulation")
    print("7. Glitch Effect")
    print("8. Voronoi Pixel Sort")
    print("9. Concentric Shapes")
    print("10. Color Shift Expansion")
    print("11. Perlin Noise Displacement")
    print("12. Spiral Pixel Sort") 

    choice = input("which Xlitch? (1-12): ")

    if choice == '1':
        # Pixel Sorting (Original)
        print("Choose direction:")
        print("1. Horizontal")
        print("2. Vertical")
        dir_choice = input("Enter your choice (1-2): ")
        if dir_choice not in ['1', '2']:
            print("Invalid direction choice.")
            return
        direction = "horizontal" if dir_choice == '1' else "vertical"

        print("Choose sorting property:")
        print("1. Color")
        print("2. Brightness")
        print("3. Hue")
        sort_choice = input("Enter your choice (1-3): ")
        if sort_choice not in ['1', '2', '3']:
            print("Invalid sorting property choice.")
            return
        sort_property = {
            '1': 'color',
            '2': 'brightness',
            '3': 'hue'
        }[sort_choice]

        chunk_size = input("Enter chunk size (width x height, e.g., 32x32): ")
        try:
            chunk_width, chunk_height = map(int, chunk_size.lower().split('x'))
        except ValueError:
            print("Invalid chunk size format.")
            return

        processed_image = pixel_sorting(image, direction, chunk_width, chunk_height, sort_property)
        effect = "pixel-sort"
        settings = f"{direction}_{chunk_width}x{chunk_height}_{sort_property}"

    elif choice == '2':
        # Pixel Sorting (Corner-to-Corner)
        print("Choose sorting property for Corner-to-Corner Sorting:")
        print("1. Color")
        print("2. Brightness")
        print("3. Hue")
        sort_choice = input("Enter your choice (1-3): ")
        if sort_choice not in ['1', '2', '3']:
            print("Invalid sorting property choice.")
            return
        sort_property = {
            '1': 'color',
            '2': 'brightness',
            '3': 'hue'
        }[sort_choice]

        chunk_size = input("Enter chunk size for Corner-to-Corner Sorting (width x height, e.g., 32x32): ")
        try:
            chunk_width, chunk_height = map(int, chunk_size.lower().split('x'))
        except ValueError:
            print("Invalid chunk size format.")
            return

        print("Choose the corner to start sorting from:")
        print("1. Top-Left")
        print("2. Top-Right")
        print("3. Bottom-Left")
        print("4. Bottom-Right")
        corner_choice = input("Enter your choice (1-4): ")
        corner_dict = {
            '1': 'top-left',
            '2': 'top-right',
            '3': 'bottom-left',
            '4': 'bottom-right'
        }
        if corner_choice not in corner_dict:
            print("Invalid corner choice.")
            return
        corner = corner_dict[corner_choice]

        print("Choose sorting direction for Corner-to-Corner Sorting:")
        print("1. Horizontal")
        print("2. Vertical")
        direction_choice = input("Enter your choice (1-2): ")
        if direction_choice not in ['1', '2']:
            print("Invalid sorting direction choice.")
            return
        horizontal = direction_choice == '1'

        processed_image = pixel_sorting_corner_to_corner(image, chunk_width, chunk_height, sort_property, corner, horizontal)
        effect = "corner-to-corner-sort"
        settings = f"{sort_property}_{chunk_width}x{chunk_height}_{corner}_{'horizontal' if horizontal else 'vertical'}"

    elif choice == '3':
        # Color Channel Manipulation
        processed_image, manipulation_description = color_channel_manipulation(image)
        effect = "color-channel-manipulation"
        settings = manipulation_description

    elif choice == '4':
        # Data Moshing (Function 4)
        print("Data Moshing will aggressively scramble the image data to create glitch effects.")

        try:
            # Intensity Input
            intensity_input = input("Enter the intensity of the effect (e.g., 10 for moderate, 20 for high): ")
            intensity = int(intensity_input)
            if intensity <= 0:
                raise ValueError("Intensity must be a positive integer.")
        except ValueError as ve:
            print(f"Invalid intensity value: {ve}")
            return

        # Movement Mode Selection
        print("\nChoose movement mode:")
        print("1. Swap Blocks (Move to New Positions)")
        print("2. Manipulate Blocks In-Place (No Movement)")
        movement_choice = input("Enter your choice (1-2): ")

        if movement_choice == '1':
            movement_mode = 'swap'
        elif movement_choice == '2':
            movement_mode = 'in_place'
        else:
            print("Invalid movement mode choice. Defaulting to 'Swap Blocks'.")
            movement_mode = 'swap'

        # Color Swap Mode Selection
        print("\nChoose color swap mode:")
        print("1. No Swap")
        print("2. Swap All Colors on Blocks")
        print("3. Randomly Swap Colors on Blocks")
        color_swap_choice = input("Enter your choice (1-3): ")

        color_swap_mode = 'random_swap'  # Default

        if color_swap_choice == '1':
            color_swap_mode = 'no_swap'
        elif color_swap_choice == '2':
            color_swap_mode = 'swap_all'
        elif color_swap_choice == '3':
            color_swap_mode = 'random_swap'
        else:
            print("Invalid color swap choice. Defaulting to 'Randomly Swap Colors on Blocks'.")
            color_swap_mode = 'random_swap'

        # Block Flip Mode Selection
        print("\nChoose block flip mode:")
        print("1. Do Not Flip Blocks")
        print("2. Flip Blocks Vertically")
        print("3. Flip Blocks Horizontally")
        print("4. Flip Blocks Randomly Vertically or Horizontally")
        block_flip_choice = input("Enter your choice (1-4): ")

        block_flip_mode = 'flip_random'  # Default

        if block_flip_choice == '1':
            block_flip_mode = 'no_flip'
        elif block_flip_choice == '2':
            block_flip_mode = 'flip_vertical'
        elif block_flip_choice == '3':
            block_flip_mode = 'flip_horizontal'
        elif block_flip_choice == '4':
            block_flip_mode = 'flip_random'
        else:
            print("Invalid block flip choice. Defaulting to 'Flip Blocks Randomly Vertically or Horizontally'.")
            block_flip_mode = 'flip_random'

        # Apply Data Moshing Effect
        processed_image = data_moshing(
            image,
            intensity=intensity,
            movement_mode=movement_mode,
            color_swap_mode=color_swap_mode,
            block_flip_mode=block_flip_mode
        )
        effect = "data-mosh"
        settings = f"block-move_{movement_mode}_intensity_{intensity}_color_swap_{color_swap_mode}_block_flip_{block_flip_mode}"

    elif choice == '5':
        # Pixel Drift
        direction = input("Enter drift direction (up/down/left/right): ").lower()
        if direction not in ['up', 'down', 'left', 'right']:
            print("Invalid drift direction.")
            return
        try:
            max_drift = int(input("Enter maximum drift amount (e.g., 30): "))
        except ValueError:
            print("Invalid drift amount.")
            return
        processed_image = pixel_drift(image, direction, max_drift)
        effect = "pixel-drift"
        settings = f"{direction}_{max_drift}"

    elif choice == '6':
        # Bit Manipulation
        processed_image = bit_manipulation(image)
        effect = "bit-manipulation"
        settings = "invert-bits"

    elif choice == '7':
        # Glitch Effect
        print("Applying glitch effect to the image.")
        try:
            max_thickness = int(input("Enter the maximum thickness of glitch lines (e.g., 10): "))
        except ValueError:
            print("Invalid thickness value.")
            return
        processed_image = glitch_image(image, max_thickness)
        effect = "glitch-effect"
        settings = f"pixel-shift_{max_thickness}"

    elif choice == '8':
        # Voronoi Pixel Sort
        print("Applying Voronoi Pixel Sort effect to the image.")
        try:
            num_cells = int(input("Enter the approximate number of Voronoi cells (e.g., 100): "))
            size_variation = float(input("Enter the size variation (0 to 1, e.g., 0.5): "))
            print("Choose sorting property:")
            print("1. Color")
            print("2. Brightness")
            print("3. Hue")
            sort_choice = input("Enter your choice (1-3): ")
            sort_by = {
                '1': 'color',
                '2': 'brightness',
                '3': 'hue'
            }.get(sort_choice, 'color')
            print("Choose sorting order:")
            print("1. Clockwise")
            print("2. Counter-Clockwise")
            order_choice = input("Enter your choice (1-2): ")
            sort_order = 'clockwise' if order_choice == '1' else 'counter-clockwise'
        except ValueError:
            print("Invalid input.")
            return
        processed_image = voronoi_pixel_sort(image, num_cells, size_variation, sort_by, sort_order)
        effect = "voronoi-pixel-sort"
        settings = f"{num_cells}_cells_{sort_by}_{sort_order}_variation_{size_variation}"

    elif choice == '9':
        # Concentric Shapes
        print("Generating concentric shapes from random points.")
        try:
            num_points = int(input("Enter the number of points (e.g., 5): "))
            print("Choose shape type:")
            print("1. Circle")
            print("2. Square")
            print("3. Triangle")
            print("4. Hexagon")
            shape_choice = input("Enter your choice (1-4): ")
            shape_type = {
                '1': 'circle',
                '2': 'square',
                '3': 'triangle',
                '4': 'hexagon'
            }.get(shape_choice, 'circle')
            thickness = int(input("Enter the thickness of shapes in pixels (e.g., 3): "))
            spacing = int(input("Enter the spacing between shapes in pixels (e.g., 10): "))
            rotation_angle = float(input("Enter the incremental rotation angle per shape in degrees (e.g., 0 for no rotation): "))
            darken_step = int(input("Enter the amount to darken each shape (0-255, e.g., 0 for no change): "))
            color_shift = float(input("Enter the amount to shift color hue per shape (0-360 degrees, e.g., 0 for no change): "))
        except ValueError:
            print("Invalid input.")
            return
        processed_image = concentric_shapes(image, num_points, shape_type, thickness,
                                            spacing, rotation_angle, darken_step, color_shift)
        effect = "concentric-shapes"
        settings = f"{num_points}_points_{shape_type}_thickness_{thickness}_spacing_{spacing}_rotation_{rotation_angle}_darken_{darken_step}_color_shift_{color_shift}"

    elif choice == '10':
        # Color Shift Expansion
        print("Applying color shift expansion from seed points.")
        try:
            num_points = int(input("Enter the number of seed points (e.g., 5): "))
            shift_amount = float(input("Enter the hue shift amount per unit distance (e.g., 5): "))
            print("Choose expansion type:")
            print("1. Square Expansion (includes diagonals)")
            print("2. Cross Expansion (excludes diagonals)")
            print("3. Circular Expansion (Euclidean distance)")
            pattern_choice = input("Enter your choice (1-3): ")
            if pattern_choice == '1':
                expansion_type = 'square'
            elif pattern_choice == '2':
                expansion_type = 'cross'
            elif pattern_choice == '3':
                expansion_type = 'circular'
            else:
                print("Invalid choice. Defaulting to Square Expansion.")
                expansion_type = 'square'
        except ValueError:
            print("Invalid input.")
            return
        processed_image = color_shift_expansion(image, num_points, shift_amount, expansion_type)
        effect = "color-shift-expansion"
        settings = f"{num_points}_points_shift_{shift_amount}_{expansion_type}"

    elif choice == '11':
        # Perlin Noise Displacement
        print("Applying Perlin Noise Displacement to the image.")
        try:
            scale = float(input("Enter the scale of the Perlin noise (e.g., 100): "))
            intensity = float(input("Enter the intensity of the displacement (e.g., 30): "))
            octaves = int(input("Enter the number of octaves (e.g., 6): "))
            persistence = float(input("Enter the persistence value (e.g., 0.5): "))
            lacunarity = float(input("Enter the lacunarity value (e.g., 2.0): "))
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            return

        processed_image = perlin_noise_displacement(
            image,
            scale=scale,
            intensity=intensity,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        effect = "perlin-noise-displacement"
        settings = f"scale_{scale}_intensity_{intensity}_octaves_{octaves}_persistence_{persistence}_lacunarity_{lacunarity}"

    # ... (other choices)
    elif choice == '12':
        # Spiral Pixel Sort (New Function 12)
        print("\nSpiral Pixel Sort will sort pixels within each chunk in a spiral order based on luminance.")

        try:
            # Chunk Size Input
            chunk_size_input = input("Enter the chunk size (e.g., 64): ")
            chunk_size = int(chunk_size_input)
            if chunk_size <= 0:
                raise ValueError("Chunk size must be a positive integer.")
        except ValueError as ve:
            print(f"Invalid chunk size value: {ve}")
            return

        # Sorting Order Selection
        print("\nChoose sorting order based on luminance:")
        print("1. Darkest to Lightest")
        print("2. Lightest to Darkest")
        order_choice = input("Enter your choice (1-2): ")

        if order_choice == '1':
            order = 'darkest-to-lightest'
        elif order_choice == '2':
            order = 'lightest-to-darkest'
        else:
            print("Invalid sorting order choice. Defaulting to 'lightest-to-darkest'.")
            order = 'lightest-to-darkest'

        # Apply Spiral Pixel Sort Effect
        sorted_image = sort_pixels_in_spiral(
            image_path=file_path,
            chunk_size=chunk_size,
            order=order
        )

        if sorted_image is None:
            print("Spiral Pixel Sort failed due to an error.")
            return

        # Convert the NumPy array to PIL Image
        processed_image = Image.fromarray(sorted_image)

        effect = "spiral-pixel-sort"
        settings = f"chunk_size_{chunk_size}_order_{order}"

    else:
        print("Invalid choice.")
        return

    if processed_image is None:
        print("Image processing failed.")
        return

    output_filename = generate_output_filename(file_path, effect, settings)
    save_image(processed_image, output_filename)
    
# FUNCTIONS
# FUNCTIONS

def pixel_sorting(image, direction, chunk_width, chunk_height, sort_by):
    """
    Sorts pixels in the image based on the specified property.

    :param image: The PIL Image to process.
    :param direction: 'horizontal' or 'vertical'.
    :param chunk_width: Width of the chunks to sort.
    :param chunk_height: Height of the chunks to sort.
    :param sort_by: Property to sort by: 'color', 'brightness', or 'hue'.
    """
    def brightness(pixel):
        # Calculate perceived brightness
        return pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114

    def hue(pixel):
        # Convert RGB to HSV and return hue
        r, g, b = [value / 255.0 for value in pixel[:3]]
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h

    # Select sorting function based on sort_by parameter
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue
    }.get(sort_by, lambda p: sum(p[:3]))

    width, height = image.size
    pixels = image.load()
    sorted_image = Image.new(image.mode, image.size)

    if direction == 'horizontal':
        # Process image horizontally
        for y in range(0, height, chunk_height):
            for x in range(0, width, chunk_width):
                # Extract chunk
                chunk = []
                for xi in range(x, min(x + chunk_width, width)):
                    for yj in range(y, min(y + chunk_height, height)):
                        chunk.append(pixels[xi, yj])
                # Sort chunk
                sorted_chunk = sorted(chunk, key=sort_function)
                # Place sorted chunk back
                idx = 0
                for xi in range(x, min(x + chunk_width, width)):
                    for yj in range(y, min(y + chunk_height, height)):
                        sorted_image.putpixel((xi, yj), sorted_chunk[idx])
                        idx += 1
    else:
        # Process image vertically
        for x in range(0, width, chunk_width):
            for y in range(0, height, chunk_height):
                # Extract chunk
                chunk = []
                for yj in range(y, min(y + chunk_height, height)):
                    for xi in range(x, min(x + chunk_width, width)):
                        chunk.append(pixels[xi, yj])
                # Sort chunk
                sorted_chunk = sorted(chunk, key=sort_function)
                # Place sorted chunk back
                idx = 0
                for yj in range(y, min(y + chunk_height, height)):
                    for xi in range(x, min(x + chunk_width, width)):
                        sorted_image.putpixel((xi, yj), sorted_chunk[idx])
                        idx += 1
    return sorted_image

def pixel_sorting_corner_to_corner(image, chunk_size, sort_by, corner, horizontal):
    """
    Performs corner-to-corner pixel sorting.

    :param image: The PIL Image to process.
    :param chunk_size: Size of the chunks to sort, e.g., '32x32'.
    :param sort_by: Property to sort by: 'color', 'brightness', or 'hue'.
    :param corner: Corner to start from: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    :param horizontal: Boolean indicating sorting direction.
    """
    def brightness(pixel):
        # Calculate perceived brightness
        return pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114

    def hue(pixel):
        # Convert RGB to HSV and return hue
        r, g, b = [value / 255.0 for value in pixel[:3]]
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h

    # Select sorting function based on sort_by parameter
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue
    }.get(sort_by, lambda p: sum(p[:3]))

    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.lower().split('x'))
    pixels = image.load()
    sorted_image = Image.new(image.mode, image.size)

    # Determine the order of chunks based on the corner
    x_ranges = range(0, width, chunk_width)
    y_ranges = range(0, height, chunk_height)
    if corner in ['top-right', 'bottom-right']:
        x_ranges = list(x_ranges)[::-1]
    if corner in ['bottom-left', 'bottom-right']:
        y_ranges = list(y_ranges)[::-1]

    for y in y_ranges:
        for x in x_ranges:
            # Extract chunk
            chunk = []
            for yj in range(y, min(y + chunk_height, height)):
                for xi in range(x, min(x + chunk_width, width)):
                    chunk.append(pixels[xi, yj])

            # Sort chunk
            sorted_chunk = sorted(chunk, key=sort_function)

            # Reverse chunk if necessary
            if horizontal:
                if corner in ['bottom-left', 'bottom-right']:
                    sorted_chunk.reverse()
            else:
                if corner in ['top-right', 'bottom-right']:
                    sorted_chunk.reverse()

            # Place sorted chunk back
            idx = 0
            for yj in range(y, min(y + chunk_height, height)):
                for xi in range(x, min(x + chunk_width, width)):
                    sorted_image.putpixel((xi, yj), sorted_chunk[idx])
                    idx += 1

    return sorted_image 

def color_channel_manipulation(image):
    """
    Manipulates color channels of the image based on user input.

    :param image: The PIL Image to process.
    :return: The manipulated image and a description of the manipulation.
    """
    print("Choose the color channel manipulation:")
    print("1. Swap Channels")
    print("2. Invert Channel")
    print("3. Adjust Channel Intensity")
    manipulation_choice = input("Enter your choice (1-3): ")

    r, g, b = image.split()
    manipulation_description = "default"

    if manipulation_choice == '1':
        # Swap channels
        print("Choose channels to swap:")
        print("1. Red-Green")
        print("2. Red-Blue")
        print("3. Green-Blue")
        swap_choice = input("Enter your choice (1-3): ")
        if swap_choice == '1':
            image = Image.merge(image.mode, (g, r, b))
            manipulation_description = "swap-red-green"
        elif swap_choice == '2':
            image = Image.merge(image.mode, (b, g, r))
            manipulation_description = "swap-red-blue"
        elif swap_choice == '3':
            image = Image.merge(image.mode, (r, b, g))
            manipulation_description = "swap-green-blue"
        else:
            print("Invalid choice.")
            manipulation_description = "invalid-swap"

    elif manipulation_choice == '2':
        # Invert a specific channel
        print("Choose a channel to invert:")
        print("1. Red")
        print("2. Green")
        print("3. Blue")
        invert_choice = input("Enter your choice (1-3): ")
        if invert_choice == '1':
            r = r.point(lambda i: 255 - i)
            manipulation_description = "invert-red"
        elif invert_choice == '2':
            g = g.point(lambda i: 255 - i)
            manipulation_description = "invert-green"
        elif invert_choice == '3':
            b = b.point(lambda i: 255 - i)
            manipulation_description = "invert-blue"
        else:
            print("Invalid choice.")
            manipulation_description = "invalid-invert"
        image = Image.merge(image.mode, (r, g, b))
    elif manipulation_choice == '3':
        # Adjust the intensity of a channel
        print("Choose a channel to adjust intensity:")
        print("1. Red")
        print("2. Green")
        print("3. Blue")
        intensity_choice = input("Enter your choice (1-3): ")
        factor = float(input("Enter the intensity factor (e.g., 1.5 for 50% increase): "))
        if intensity_choice == '1':
            r = r.point(lambda i: min(255, int(i * factor)))
            manipulation_description = f"adjust-red-{factor}"
        elif intensity_choice == '2':
            g = g.point(lambda i: min(255, int(i * factor)))
            manipulation_description = f"adjust-green-{factor}"
        elif intensity_choice == '3':
            b = b.point(lambda i: min(255, int(i * factor)))
            manipulation_description = f"adjust-blue-{factor}"
        else:
            print("Invalid choice.")
            manipulation_description = "invalid-adjust"
        image = Image.merge(image.mode, (r, g, b))
    else:
        print("Invalid manipulation choice.")
        manipulation_description = "invalid-choice"
    return image, manipulation_description

def data_moshing(image, intensity=10, movement_mode='swap', color_swap_mode='random_swap', block_flip_mode='random_flip'):
    """
    Applies a data moshing effect by either swapping blocks of pixels or manipulating them in place,
    with options for color channel swapping and block flipping.

    :param image: The PIL Image to process.
    :param intensity: The intensity of the effect (higher values produce more pronounced effects).
    :param movement_mode: Determines block movement behavior ('swap', 'in_place').
    :param color_swap_mode: Determines color swapping behavior ('no_swap', 'swap_all', 'random_swap').
    :param block_flip_mode: Determines block flipping behavior ('no_flip', 'flip_vertical', 'flip_horizontal', 'flip_random').
    :return: The moshed image.
    """
    import numpy as np

    image_np = np.array(image)
    height, width, channels = image_np.shape

    # To keep track of swapped block positions to avoid overlapping swaps
    swapped_positions = set()

    for _ in range(intensity * 10):
        # Determine block size based on intensity
        max_block_size = max(1, min(height, width) // (intensity // 2 or 1))

        # Random block size for each iteration
        block_size = random.randint(1, max_block_size)

        # Randomly select the first block
        x1 = random.randint(0, width - block_size)
        y1 = random.randint(0, height - block_size)

        if movement_mode == 'swap':
            # Randomly select the second block ensuring it's not the same as the first
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                x2 = random.randint(0, width - block_size)
                y2 = random.randint(0, height - block_size)
                if (x2, y2) != (x1, y1) and (x2, y2) not in swapped_positions:
                    break
                attempts += 1
            else:
                # If suitable second block not found, skip swapping
                continue

            # Swap the blocks
            block1 = np.copy(image_np[y1:y1 + block_size, x1:x1 + block_size, :])
            block2 = np.copy(image_np[y2:y2 + block_size, x2:x2 + block_size, :])
            image_np[y1:y1 + block_size, x1:x1 + block_size, :] = block2
            image_np[y2:y2 + block_size, x2:x2 + block_size, :] = block1

            # Mark these positions as swapped to avoid immediate reselection
            swapped_positions.add((x1, y1))
            swapped_positions.add((x2, y2))

            # Determine if color swapping should be applied based on color_swap_mode
            swap_colors = False
            if color_swap_mode == 'swap_all':
                swap_colors = True
            elif color_swap_mode == 'random_swap':
                swap_colors = random.choice([True, False])  # Randomly decide per swap

            if swap_colors:
                # Randomly choose a color channel swap
                swap_choice = random.choice(['red-green', 'red-blue', 'green-blue'])

                if swap_choice == 'red-green':
                    # Swap red and green channels in both blocks
                    image_np[y1:y1 + block_size, x1:x1 + block_size, [0, 1]] = image_np[y1:y1 + block_size, x1:x1 + block_size, [1, 0]]
                    image_np[y2:y2 + block_size, x2:x2 + block_size, [0, 1]] = image_np[y2:y2 + block_size, x2:x2 + block_size, [1, 0]]
                elif swap_choice == 'red-blue':
                    # Swap red and blue channels in both blocks
                    image_np[y1:y1 + block_size, x1:x1 + block_size, [0, 2]] = image_np[y1:y1 + block_size, x1:x1 + block_size, [2, 0]]
                    image_np[y2:y2 + block_size, x2:x2 + block_size, [0, 2]] = image_np[y2:y2 + block_size, x2:x2 + block_size, [2, 0]]
                elif swap_choice == 'green-blue':
                    # Swap green and blue channels in both blocks
                    image_np[y1:y1 + block_size, x1:x1 + block_size, [1, 2]] = image_np[y1:y1 + block_size, x1:x1 + block_size, [2, 1]]
                    image_np[y2:y2 + block_size, x2:x2 + block_size, [1, 2]] = image_np[y2:y2 + block_size, x2:x2 + block_size, [2, 1]]

            # Determine if block flipping should be applied based on block_flip_mode
            flip_mode = block_flip_mode
            if flip_mode == 'flip_random':
                flip_mode = random.choice(['flip_vertical', 'flip_horizontal'])

            if flip_mode == 'flip_vertical':
                # Flip both blocks vertically
                image_np[y1:y1 + block_size, x1:x1 + block_size, :] = np.flipud(image_np[y1:y1 + block_size, x1:x1 + block_size, :])
                image_np[y2:y2 + block_size, x2:x2 + block_size, :] = np.flipud(image_np[y2:y2 + block_size, x2:x2 + block_size, :])
            elif flip_mode == 'flip_horizontal':
                # Flip both blocks horizontally
                image_np[y1:y1 + block_size, x1:x1 + block_size, :] = np.fliplr(image_np[y1:y1 + block_size, x1:x1 + block_size, :])
                image_np[y2:y2 + block_size, x2:x2 + block_size, :] = np.fliplr(image_np[y2:y2 + block_size, x2:x2 + block_size, :])
            # 'no_flip' requires no action

        elif movement_mode == 'in_place':
            # Manipulate blocks in place without moving
            # Select one block to manipulate
            block = image_np[y1:y1 + block_size, x1:x1 + block_size, :]

            # Determine if color swapping should be applied based on color_swap_mode
            swap_colors = False
            if color_swap_mode == 'swap_all':
                swap_colors = True
            elif color_swap_mode == 'random_swap':
                swap_colors = random.choice([True, False])  # Randomly decide per block

            if swap_colors:
                # Randomly choose a color channel swap
                swap_choice = random.choice(['red-green', 'red-blue', 'green-blue'])

                if swap_choice == 'red-green':
                    # Swap red and green channels
                    image_np[y1:y1 + block_size, x1:x1 + block_size, [0, 1]] = block[:, :, [1, 0]]
                elif swap_choice == 'red-blue':
                    # Swap red and blue channels
                    image_np[y1:y1 + block_size, x1:x1 + block_size, [0, 2]] = block[:, :, [2, 0]]
                elif swap_choice == 'green-blue':
                    # Swap green and blue channels
                    image_np[y1:y1 + block_size, x1:x1 + block_size, [1, 2]] = block[:, :, [2, 1]]

            # Determine if block flipping should be applied based on block_flip_mode
            flip_mode = block_flip_mode
            if flip_mode == 'flip_random':
                flip_mode = random.choice(['flip_vertical', 'flip_horizontal'])

            if flip_mode == 'flip_vertical':
                # Flip block vertically
                image_np[y1:y1 + block_size, x1:x1 + block_size, :] = np.flipud(block)
            elif flip_mode == 'flip_horizontal':
                # Flip block horizontally
                image_np[y1:y1 + block_size, x1:x1 + block_size, :] = np.fliplr(block)
            # 'no_flip' requires no action

    moshed_image = Image.fromarray(image_np)
    return moshed_image

def pixel_drift(image, direction='down', max_drift=30):
    """
    Creates a pixel drift effect by shifting pixels in a specified direction,
    with random drift amounts for a more pronounced effect.

    :param image: The PIL Image to process.
    :param direction: The direction to drift pixels ('up', 'down', 'left', 'right').
    :param max_drift: Maximum drift amount in pixels.
    :return: The drifted image.
    """
    pixels = image.load()
    width, height = image.size
    drifted_image = Image.new(image.mode, image.size)

    if direction in ['down', 'up']:
        # Vertical drift with random amounts per column
        for x in range(width):
            drift_amount = random.randint(0, max_drift)
            if direction == 'up':
                drift_amount = -drift_amount
            for y in range(height):
                y_new = (y + drift_amount) % height
                drifted_image.putpixel((x, y_new), pixels[x, y])
    elif direction in ['left', 'right']:
        # Horizontal drift with random amounts per row
        for y in range(height):
            drift_amount = random.randint(0, max_drift)
            if direction == 'left':
                drift_amount = -drift_amount
            for x in range(width):
                x_new = (x + drift_amount) % width
                drifted_image.putpixel((x_new, y), pixels[x, y])
    else:
        print("Invalid direction.")
        return image

    return drifted_image

def bit_manipulation(image):
    """
    Manipulates the bits of the image data to create a glitch effect.

    :param image: The PIL Image to process.
    :return: The manipulated image.
    """
    # Convert image to bytes
    image_bytes = bytearray(image.tobytes())
    # Invert every other byte
    manipulated_bytes = bytearray(b ^ 0xFF if i % 2 == 0 else b for i, b in enumerate(image_bytes))
    # Reconstruct image from bytes
    manipulated_image = Image.frombytes(image.mode, image.size, bytes(manipulated_bytes))
    return manipulated_image

def glitch_image(image, max_thickness=10):
    """
    Applies a glitch effect by shifting blocks of rows and columns of pixels.

    :param image: The PIL Image to process.
    :param max_thickness: The maximum thickness of the glitch lines.
    :return: The glitched image.
    """
    import numpy as np

    image_np = np.array(image)
    height, width, channels = image_np.shape

    # Randomly decide the number of glitch operations
    num_glitches = random.randint(10, 30)
    for _ in range(num_glitches):
        # Randomly choose whether to glitch rows or columns
        if random.choice([True, False]):
            # Glitch a block of rows
            thickness = random.randint(1, max_thickness)
            y = random.randint(0, height - thickness)
            shift = random.randint(-width // 2, width // 2)
            image_np[y:y+thickness, :] = np.roll(image_np[y:y+thickness, :], shift, axis=1)
        else:
            # Glitch a block of columns
            thickness = random.randint(1, max_thickness)
            x = random.randint(0, width - thickness)
            shift = random.randint(-height // 2, height // 2)
            image_np[:, x:x+thickness, :] = np.roll(image_np[:, x:x+thickness, :], shift, axis=0)

    glitched_image = Image.fromarray(image_np)
    return glitched_image

def voronoi_pixel_sort(image, num_cells=100, size_variation=0.5, sort_by='color', sort_order='clockwise'):
    """
    Applies a Voronoi-based pixel sorting effect to the image.

    :param image: The PIL Image to process.
    :param num_cells: Approximate number of Voronoi cells.
    :param size_variation: Variability in cell sizes (0 to 1).
    :param sort_by: Property to sort by ('color', 'brightness', 'hue').
    :param sort_order: Sorting order ('clockwise' or 'counter-clockwise').
    :return: The processed image.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    # Convert image to NumPy array
    image_np = np.array(image)
    height, width, channels = image_np.shape

    # Generate seed points for Voronoi cells
    num_points = num_cells
    # Adjust for size variation
    variation = int(num_points * size_variation)
    point_counts = num_points + random.randint(-variation, variation)
    xs = np.random.randint(0, width, size=point_counts)
    ys = np.random.randint(0, height, size=point_counts)
    points = np.vstack((xs, ys)).T

    # Assign each pixel to the nearest seed point
    # Create a grid of pixel coordinates
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.column_stack((xv.ravel(), yv.ravel()))

    # Build a KD-Tree for efficient nearest-neighbor search
    tree = cKDTree(points)
    _, regions = tree.query(pixel_coords)

    # Reshape regions to the image shape
    regions = regions.reshape((height, width))

    # Define sorting functions
    def brightness(pixel):
        return pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114

    def hue(pixel):
        r, g, b = [value / 255.0 for value in pixel[:3]]
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h

    sort_function = {
        'color': lambda p: np.sum(p[:3], dtype=int),
        'brightness': brightness,
        'hue': hue
    }.get(sort_by, lambda p: np.sum(p[:3], dtype=int))

    # Process each cell
    for region_id in np.unique(regions):
        # Get the mask for the current region
        mask = regions == region_id

        # Get the coordinates of pixels in the cell
        ys_cell, xs_cell = np.where(mask)

        # Get the pixels in the cell
        pixels = image_np[ys_cell, xs_cell]

        # Compute the centroid of the cell
        centroid_x = np.mean(xs_cell)
        centroid_y = np.mean(ys_cell)

        # Calculate the angle of each pixel relative to the centroid
        angles = np.arctan2(ys_cell - centroid_y, xs_cell - centroid_x)
        if sort_order == 'clockwise':
            angles = -angles  # Reverse the angle for clockwise sorting

        # Get indices that would sort positions by angle
        position_order = np.argsort(angles)

        # Sort the pixels based on the sorting property
        pixel_values = np.array([sort_function(p) for p in pixels])
        pixel_order = np.argsort(pixel_values)

        # Rearrange the pixels and positions
        sorted_pixels = pixels[pixel_order]
        sorted_ys = ys_cell[position_order]
        sorted_xs = xs_cell[position_order]

        # Assign sorted pixels to positions sorted by angle
        image_np[sorted_ys, sorted_xs] = sorted_pixels

    # Convert back to PIL Image
    processed_image = Image.fromarray(image_np)
    return processed_image

def concentric_shapes(image, num_points=5, shape_type='circle', thickness=3, spacing=10,
                      rotation_angle=0, darken_step=0, color_shift=0):
    """
    Generates concentric shapes from random points in the image.

    :param image: The PIL Image to process.
    :param num_points: Number of random pixels to select.
    :param shape_type: Type of shape ('square', 'circle', 'hexagon', 'triangle').
    :param thickness: Thickness of the shapes in pixels.
    :param spacing: Spacing between shapes in pixels.
    :param rotation_angle: Incremental rotation angle in degrees for each subsequent shape.
    :param darken_step: Amount to darken the color for each subsequent shape (0-255).
    :param color_shift: Amount to shift the hue for each shape (0-360 degrees).
    :return: The processed image.
    """
    from PIL import ImageDraw

    width, height = image.size
    image = image.convert('RGBA')  # Ensure image has an alpha channel

    # Create a base image to draw on
    base_image = image.copy()

    # Select random points
    xs = np.random.randint(0, width, size=num_points)
    ys = np.random.randint(0, height, size=num_points)
    points = list(zip(xs, ys))

    # For each point
    for x0, y0 in points:
        # Get the color of the pixel
        original_color = base_image.getpixel((x0, y0))

        # Initialize variables
        current_size = spacing
        current_rotation = 0  # Initialize cumulative rotation

        # Initialize HSV color from the original color
        r, g, b = original_color[:3]
        h_original, s_original, v_original = rgb_to_hsv(r, g, b)
        current_hue = h_original  # Start with the original hue

        max_dimension = max(width, height) * 1.5  # Set a maximum size to prevent infinite loops

        while current_size < max_dimension:
            # Adjust the hue for the current shape
            if color_shift != 0:
                current_hue = (current_hue + color_shift) % 360
            # Convert HSV back to RGB
            current_color = hsv_to_rgb(current_hue, s_original, v_original)

            # Darken the color if darken_step is set
            if darken_step != 0:
                current_color = darken_color(current_color, darken_step)

            # Create a shape image to draw the shape
            shape_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            shape_draw = ImageDraw.Draw(shape_image)

            # Calculate the points of the shape
            if shape_type == 'circle':
                bbox = [x0 - current_size, y0 - current_size, x0 + current_size, y0 + current_size]
                shape_draw.ellipse(bbox, outline=current_color, width=thickness)
                shape_bbox = bbox
            else:
                if shape_type == 'square':
                    half_size = current_size
                    points_list = [
                        (x0 - half_size, y0 - half_size),
                        (x0 + half_size, y0 - half_size),
                        (x0 + half_size, y0 + half_size),
                        (x0 - half_size, y0 + half_size)
                    ]
                elif shape_type == 'triangle':
                    half_size = current_size
                    height_triangle = half_size * math.sqrt(3)
                    points_list = [
                        (x0, y0 - 2 * half_size / math.sqrt(3)),
                        (x0 - half_size, y0 + height_triangle / 3),
                        (x0 + half_size, y0 + height_triangle / 3)
                    ]
                elif shape_type == 'hexagon':
                    half_size = current_size
                    points_list = []
                    for angle in range(0, 360, 60):
                        rad = math.radians(angle + current_rotation)
                        px = x0 + half_size * math.cos(rad)
                        py = y0 + half_size * math.sin(rad)
                        points_list.append((px, py))
                else:
                    print(f"Unsupported shape type: {shape_type}")
                    return base_image.convert('RGB')

                # Apply cumulative rotation
                if current_rotation != 0 and shape_type != 'hexagon':
                    points_list = rotate_points(points_list, (x0, y0), current_rotation)

                # Draw the shape
                shape_draw.polygon(points_list, outline=current_color, width=thickness)

                # Calculate the bounding box of the shape
                xs_list = [p[0] for p in points_list]
                ys_list = [p[1] for p in points_list]
                shape_bbox = [min(xs_list), min(ys_list), max(xs_list), max(ys_list)]

            # Check if the shape is completely outside the image bounds
            if (shape_bbox[2] < 0 or shape_bbox[0] > width or
                    shape_bbox[3] < 0 or shape_bbox[1] > height):
                break

            # Composite the shape onto the base image
            base_image = Image.alpha_composite(base_image, shape_image)

            # Update variables
            current_size += spacing + thickness

            # Increment the rotation angle
            current_rotation += rotation_angle

        # No need to update previous_image or use ImageChops.difference

    return base_image.convert('RGB')

def rotate_points(points, center, angle_degrees):
    """
    Rotates a list of points around a center point by a given angle.

    :param points: List of (x, y) tuples.
    :param center: The center point (x0, y0) around which to rotate.
    :param angle_degrees: The angle in degrees to rotate.
    :return: List of rotated (x, y) tuples.
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    x0, y0 = center
    rotated_points = []
    for x, y in points:
        x_shifted = x - x0
        y_shifted = y - y0
        x_rotated = x_shifted * cos_theta - y_shifted * sin_theta + x0
        y_rotated = x_shifted * sin_theta + y_shifted * cos_theta + y0
        rotated_points.append((x_rotated, y_rotated))
    return rotated_points

def darken_color(color, darken_step):
    """
    Darkens the given RGB color by the specified amount.

    :param color: Tuple of (R, G, B)
    :param darken_step: Amount to darken (0-255)
    :return: Darkened color tuple (R, G, B)
    """
    r, g, b = color[:3]
    r = max(0, r - darken_step)
    g = max(0, g - darken_step)
    b = max(0, b - darken_step)
    return (r, g, b)

def rgb_to_hsv(r, g, b):
    """
    Converts RGB color to HSV.
    Input RGB values should be in the range [0, 255].
    Returns HSV values with h in degrees [0, 360), s and v in [0, 1].
    """
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h_degrees = h * 360
    return h_degrees, s, v

def hsv_to_rgb(h, s, v):
    """
    Converts HSV color to RGB.
    Input h in degrees [0, 360), s and v in [0, 1].
    Returns RGB values in the range [0, 255].
    """
    h_norm = h / 360.0
    r_norm, g_norm, b_norm = colorsys.hsv_to_rgb(h_norm, s, v)
    r, g, b = int(r_norm * 255), int(g_norm * 255), int(b_norm * 255)
    return (r, g, b)

def color_shift_expansion(image, num_points=5, shift_amount=5, expansion_type='square'):
    """
    Expands color shifts from seed points across the image, shifting the existing pixel colors.

    :param image: The PIL Image to process.
    :param num_points: Number of seed points to generate.
    :param shift_amount: Amount to shift the hue per pixel distance.
    :param expansion_type: Type of expansion ('square', 'cross', 'circular').
    :return: The processed image.
    """
    import heapq

    width, height = image.size
    image = image.convert('RGB')
    image_np = np.array(image)

    # Convert image to HSV
    image_hsv = np.zeros((height, width, 3), dtype=float)
    for y in range(height):
        for x in range(width):
            r, g, b = image_np[y, x]
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            image_hsv[y, x] = [h * 360, s, v]

    # Initialize assigned pixels mask
    assigned = np.zeros((height, width), dtype=bool)

    # Generate seed points
    xs = np.random.randint(0, width, size=num_points)
    ys = np.random.randint(0, height, size=num_points)
    seed_points = list(zip(xs, ys))

    # For each seed point, store shift direction and initialize queue
    seed_info = []
    for idx, (x0, y0) in enumerate(seed_points):
        shift_direction = random.choice(['add', 'subtract'])
        seed_info.append({
            'position': (x0, y0),
            'shift_direction': shift_direction,
            'queue': [],
            'distance_map': np.full((height, width), np.inf),
        })
        # Mark the seed point as assigned
        assigned[y0, x0] = True
        # Distance from seed point to itself is 0
        seed_info[idx]['distance_map'][y0, x0] = 0
        # Enqueue the seed point with priority (distance)
        if expansion_type == 'circular':
            heapq.heappush(seed_info[idx]['queue'], (0, x0, y0))
        else:
            seed_info[idx]['queue'].append((x0, y0))

    # Define neighbor offsets based on expansion_type
    if expansion_type == 'square':
        # 8-connected neighbors
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]
    elif expansion_type == 'cross':
        # 4-connected neighbors
        neighbor_offsets = [(-1, 0),
                            (0, -1),        (0, 1),
                            (1, 0)]
    elif expansion_type == 'circular':
        # All possible neighbors, but distance will determine expansion
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]
    else:
        print(f"Invalid expansion type: {expansion_type}. Defaulting to square.")
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]

    # Continue expansion until all pixels are assigned
    while any(info['queue'] for info in seed_info):
        for info in seed_info:
            queue = info['queue']
            if not queue:
                continue  # No more pixels to process for this seed point

            if expansion_type == 'circular':
                # Pop the pixel with the smallest distance
                current_distance, x, y = heapq.heappop(queue)
            else:
                x, y = queue.pop(0)
                current_distance = info['distance_map'][y, x]

            # Expand to neighbors
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not assigned[ny, nx]:
                    assigned[ny, nx] = True
                    # Calculate Euclidean distance from seed point
                    if expansion_type == 'circular':
                        x0, y0 = info['position']
                        new_distance = np.hypot(nx - x0, ny - y0)
                    else:
                        new_distance = current_distance + 1
                    info['distance_map'][ny, nx] = new_distance
                    # Get existing pixel's HSV
                    existing_h, existing_s, existing_v = image_hsv[ny, nx]
                    # Calculate hue shift
                    hue_shift = shift_amount * new_distance
                    # Shift existing hue
                    if info['shift_direction'] == 'add':
                        new_hue = (existing_h + hue_shift) % 360
                    else:
                        new_hue = (existing_h - hue_shift) % 360
                    # Update HSV value
                    image_hsv[ny, nx] = [new_hue, existing_s, existing_v]
                    # Enqueue the neighbor
                    if expansion_type == 'circular':
                        heapq.heappush(queue, (new_distance, nx, ny))
                    else:
                        queue.append((nx, ny))

    # Convert the modified HSV image back to RGB
    for y in range(height):
        for x in range(width):
            h, s, v = image_hsv[y, x]
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
            image_np[y, x] = [int(r * 255), int(g * 255), int(b * 255)]

    # Convert back to PIL Image
    processed_image = Image.fromarray(image_np)
    return processed_image

def perlin_noise_displacement(image, scale=100, intensity=30, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Applies a Perlin noise-based displacement to the image.

    :param image: PIL Image to process.
    :param scale: Scale of the Perlin noise.
    :param intensity: Maximum displacement in pixels.
    :param octaves: Number of layers of noise.
    :param persistence: Amplitude of each octave.
    :param lacunarity: Frequency of each octave.
    :return: Displaced PIL Image.
    """
    width, height = image.size
    image_np = np.array(image)
    displaced_image = np.zeros_like(image_np)

    # Generate Perlin noise for both x and y displacements
    perlin_x = np.zeros((height, width))
    perlin_y = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            perlin_x[y][x] = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0
            )
            perlin_y[y][x] = noise.pnoise2(
                (x + 100) / scale,  # Offset to generate different noise
                (y + 100) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0
            )

    # Normalize noise to range [-1, 1]
    perlin_x = perlin_x / np.max(np.abs(perlin_x))
    perlin_y = perlin_y / np.max(np.abs(perlin_y))

    # Apply displacement
    # Vectorized approach for better performance
    displacement_x = (perlin_x * intensity).astype(int)
    displacement_y = (perlin_y * intensity).astype(int)

    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate source coordinates with displacement
    src_x = np.clip(x_coords + displacement_x, 0, width - 1)
    src_y = np.clip(y_coords + displacement_y, 0, height - 1)

    # Apply displacement
    displaced_image = image_np[src_y, src_x]

    return Image.fromarray(displaced_image)

def spiral_coords(n):
    """
    Generates spiral coordinates for an n x n matrix in spiral order.

    :param n: Size of the square matrix.
    :return: List of (row, col) tuples in spiral order.
    """
    coords = []
    left, right = 0, n - 1
    top, bottom = 0, n - 1

    while left <= right and top <= bottom:
        # Traverse from left to right
        for col in range(left, right + 1):
            coords.append((top, col))
        top += 1

        # Traverse downwards
        for row in range(top, bottom + 1):
            coords.append((row, right))
        right -= 1

        if top <= bottom:
            # Traverse from right to left
            for col in range(right, left - 1, -1):
                coords.append((bottom, col))
            bottom -= 1

        if left <= right:
            # Traverse upwards
            for row in range(bottom, top - 1, -1):
                coords.append((row, left))
            left += 1

    return coords

def sort_pixels_in_spiral(image_path, chunk_size, order):
    """
    Applies a spiral pixel sort effect to an image.

    :param image_path: Path to the input image.
    :param chunk_size: Size of the square chunks to divide the image into.
    :param order: Sorting order based on luminance ('darkest-to-lightest' or 'lightest-to-darkest').
    :return: Sorted image as a NumPy array.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Get image dimensions
    height, width, channels = image.shape

    # Calculate number of chunks along height and width
    num_chunks_y = height // chunk_size
    num_chunks_x = width // chunk_size

    # Pad the image if necessary to make it divisible by chunk_size
    pad_y = (chunk_size - (height % chunk_size)) % chunk_size
    pad_x = (chunk_size - (width % chunk_size)) % chunk_size

    if pad_y != 0 or pad_x != 0:
        image = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
        height, width, channels = image.shape
        num_chunks_y = height // chunk_size
        num_chunks_x = width // chunk_size

    # Split the image into chunks
    chunks = []
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            chunk = image[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size]
            chunks.append(chunk)

    # Generate spiral coordinates
    spiral_order = spiral_coords(chunk_size)
    total_pixels = chunk_size * chunk_size

    # Sort each chunk
    sorted_chunks = []
    for chunk in chunks:
        # Flatten the chunk and sort based on luminance
        flattened_chunk = chunk.reshape(-1, channels)
        luminance = np.mean(flattened_chunk, axis=1)
        sorted_indices = np.argsort(luminance)

        if order == 'darkest-to-lightest':
            sorted_indices = sorted_indices[::-1]
        elif order != 'lightest-to-darkest':
            print(f"Warning: Unknown order '{order}'. Defaulting to 'lightest-to-darkest'.")
            order = 'lightest-to-darkest'

        # Assign sorted pixels to spiral coordinates
        sorted_chunk = np.zeros_like(chunk)
        for idx, (row, col) in zip(sorted_indices[:total_pixels], spiral_order):
            sorted_chunk[row, col] = flattened_chunk[idx]

        sorted_chunks.append(sorted_chunk)

    # Combine sorted chunks back into the image
    sorted_image = np.zeros_like(image)
    chunk_idx = 0
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            sorted_image[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size] = sorted_chunks[chunk_idx]
            chunk_idx += 1

    # Remove padding if it was added
    if pad_y != 0 or pad_x != 0:
        sorted_image = sorted_image[:height - pad_y, :width - pad_x]

    # Convert BGR to RGB
    sorted_image = cv2.cvtColor(sorted_image, cv2.COLOR_BGR2RGB)

    return sorted_image

if __name__ == "__main__":
    process_image()
