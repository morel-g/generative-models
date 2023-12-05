import imageio
import textwrap
from PIL import Image, ImageDraw, ImageFont
import os
from src.utils import ensure_directory_exists
from src.eval.plot_utils import get_font

FIG_DIR = "figures/"


def save_strings_to_png(
    strings: list[str],
    output_dir: str,
    name: str = "output.png",
    fig_dir: str = FIG_DIR,
    bg_color: str = "white",
    text_color: str = "black",
    padding: int = 10,
    spacing: int = 10,
    max_line_length: int = 150,
) -> None:
    """
    Save a list of strings as a PNG image.

    Parameters:
    - strings (list[str]): The strings to save.
    - output_dir (str): The directory to save the image in.
    - name (str): The name of the output image file. Defaults to "output.png".
    - fig_dir (str): The subdirectory for figures. Defaults to the global FIG_DIR.
    - bg_color (str): The background color. Defaults to "white".
    - text_color (str): The text color. Defaults to "black".
    - padding (int): The padding around the text. Defaults to 10.
    - spacing (int): The spacing between lines of text. Defaults to 10.
    - max_line_length (int): The maximum line length before wrapping text. Defaults to 150.

    Returns:
    - None
    """
    ensure_directory_exists(os.path.join(output_dir, fig_dir))
    if not name.endswith(".png"):
        name += ".png"
    font = get_font(size=12)

    # Dummy image for text size calculation
    dummy_img = Image.new("RGB", (1, 1), bg_color)
    dummy_draw = ImageDraw.Draw(dummy_img)

    # Calculate the dimensions of the text block
    text_width = 0
    blocks = []

    for i, string in enumerate(strings, start=1):
        # Calculate space required for "Sample X:"
        sample_header = f"Sample {i}:"
        header_width, header_height = dummy_draw.textsize(sample_header, font=font)
        text_width = max(text_width, header_width + padding * 2)
        blocks.append((sample_header, header_height + spacing))

        # Process each line in the string
        for paragraph in string.split("\n"):
            for line in textwrap.wrap(paragraph, max_line_length):
                line_width, line_height = dummy_draw.textsize(line, font=font)
                text_width = max(text_width, line_width + padding * 2)
                blocks.append((line, line_height))

        # Add space for separator line
        blocks.append(("separator", spacing))

    text_height = sum(height for _, height in blocks) + padding
    image = Image.new("RGB", (text_width, text_height), bg_color)
    draw = ImageDraw.Draw(image)

    # Draw text on the image
    y = padding
    for i, (text, height) in enumerate(blocks):
        if text == "separator":
            y += height
            continue

        if text.startswith("Sample"):
            # Draw line above sample header
            draw.line(
                [
                    (padding, y - spacing / 2),
                    (text_width - padding, y - spacing / 2),
                ],
                fill=text_color,
            )

        draw.text((padding, y), text, font=font, fill=text_color)

        if text.startswith("Sample"):
            # Adjust y-coordinate for Sample header
            y += height - spacing  # Adjusted height for the sample header
            # Draw line below sample header
            draw.line([(padding, y), (text_width - padding, y)], fill=text_color)
            y += spacing
        else:
            y += height

    # Save image
    image.save(os.path.join(output_dir, fig_dir, name))


def save_text_animation(
    strings: list[str],
    output_dir: str,
    name: str = "animation.gif",
    titles: list[str] = None,
    fig_dir: str = "figures",
    text_color: str = "black",
    bg_color: str = "white",
    duration: float = 500.0,
    font_size: int = 12,
    max_line_length: int = 150,
) -> None:
    """
    Create and save a GIF animation from a list of strings with titles.

    Parameters:
    - strings (list[str]): List of strings for the animation.
    - output_dir (str): Directory to save the GIF in.
    - name (str): Name of the output GIF file. Defaults to "animation.gif".
    - titles (list[str]): List of titles for each frame. If None, no titles will be added.
    - text_color (str): Text color. Defaults to "black".
    - bg_color (str): Background color. Defaults to "white".
    - duration (float): Duration of each frame in the animation. Defaults to 500.
    - title_font_size (int): Font size for the titles. Defaults to 14.
    - fig_dir (str): Subdirectory for figures. Defaults to "figures".
    - max_line_length (int): Maximum number of characters in one line. Defaults to 150.

    Returns:
    - None
    """
    ensure_directory_exists(os.path.join(output_dir, fig_dir))
    if not name.endswith(".gif"):
        name += ".gif"

    font = get_font(size=12)

    wrapped_strings = [textwrap.fill(text, width=max_line_length) for text in strings]
    if titles is not None:
        wrapped_titles = [
            textwrap.fill(title, width=max_line_length) for title in titles
        ]
        max_title_size = max(
            [
                ImageDraw.Draw(Image.new("RGB", (1, 1))).textsize(title, font=font)
                for title in wrapped_titles
            ],
            default=(0, 0),
        )
    else:
        max_title_size = (0, 0)
        wrapped_titles = [None] * len(wrapped_strings)

    # Calculate image width and height based on the largest string
    max_text_size = max(
        [
            ImageDraw.Draw(Image.new("RGB", (1, 1))).multiline_textsize(text, font=font)
            for text in wrapped_strings
        ],
        default=(0, 0),
    )
    image_width = max(max_text_size[0], max_title_size[0]) + 10  # Adding 10 for padding
    image_height = (
        max_text_size[1] + max_title_size[1] + 15
    )  # Adding 15 for padding between title and text
    image_size = (image_width, image_height)

    image_files = []
    try:
        title_height_with_padding = (
            max_title_size[1] + 10
        )  # 10 pixels padding below the title

        for i, (text, title) in enumerate(zip(wrapped_strings, wrapped_titles)):
            image = Image.new("RGB", image_size, bg_color)
            draw = ImageDraw.Draw(image)

            text_y_position = title_height_with_padding
            if title is not None:
                title_size = draw.textsize(title, font=font)
                title_position = ((image_size[0] - title_size[0]) / 2, 5)
                draw.text(title_position, title, font=font, fill=text_color)
                text_y_position += 5  # Optional: additional padding below the title

            text_size = draw.multiline_textsize(text, font=font)
            text_position = (
                (image_size[0] - text_size[0]) / 2,
                text_y_position,
            )
            draw.multiline_text(text_position, text, font=font, fill=text_color)

            image_file = os.path.join(output_dir, f"frame_{i:02d}.png")
            image.save(image_file)
            image_files.append(image_file)

        # Create GIF
        images = [imageio.imread(image_file) for image_file in image_files]
        file_path = os.path.join(output_dir, fig_dir, name)
        imageio.mimsave(file_path, images, duration=duration, loop=0)

    finally:
        # Clean up the temporary image files
        for image_file in image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
