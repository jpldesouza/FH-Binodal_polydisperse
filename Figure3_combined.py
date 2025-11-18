import pymupdf as fitz
from PIL import Image, ImageDraw


def stack_pdfs_vertically(
    pdf_paths,
    output_pdf="Figure3_combined.pdf",
    dpi=300,
    line_width_pts=2
):
    """
    Stack multiple single-page PDFs vertically into one PDF, with
    a horizontal black separator line between each.

    Parameters
    ----------
    pdf_paths : list of str
        List of input PDF file paths (each assumed single-page).
    output_pdf : str
        Output PDF filename.
    dpi : int
        Rendering resolution for rasterization.
    line_width_pts : float
        Separator line thickness in PDF points (1/72 inch).
    """

    # Convert line width from points to pixels
    line_width_px = max(1, int(round(line_width_pts * dpi / 72.0)))

    # Render each PDF page to a PIL image
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images = []
    widths = []
    heights = []

    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
        widths.append(img.width)
        heights.append(img.height)
        doc.close()

    # Canvas dimensions: max width, sum of heights + separators
    max_w = max(widths)
    total_h = sum(heights) + line_width_px * (len(images) - 1)

    # Create output canvas (white background)
    out_img = Image.new("RGB", (max_w, total_h), "white")
    draw = ImageDraw.Draw(out_img)

    # Paste images and draw horizontal lines between them
    y_offset = 0
    for i, img in enumerate(images):
        # Center each image horizontally if widths differ
        x_offset = (max_w - img.width) // 2
        out_img.paste(img, (x_offset, y_offset))

        y_offset += img.height

        # Draw separator line except after the last image
        if i < len(images) - 1:
            draw.rectangle(
                [0, y_offset, max_w, y_offset + line_width_px - 1],
                fill="black"
            )
            y_offset += line_width_px

    # Save as PDF
    out_img.save(output_pdf, "PDF", resolution=dpi)
    print(f"✓ Saved stacked PDF as {output_pdf}")


# ----------------------------------------------------------------------
# Example usage: stack Figure3ad, Figure3eh, Figure3il in a vertical column
# ----------------------------------------------------------------------
stack_pdfs_vertically(
    [
        "Figure3ad_final.pdf",
        "Figure3eh_final.pdf",
        "Figure3il_final.pdf",
    ],
    output_pdf="Figure3_final.pdf",
    dpi=300,
    line_width_pts=2,
)
