import pymupdf as fitz
from PIL import Image, ImageDraw

def combine_pdfs_side_by_side(
    pdf_left,
    pdf_right,
    output_pdf="Figure2_combined.pdf",
    dpi=300,
    line_width_pts=2
):
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Open PDFs
    doc_left = fitz.open(pdf_left)
    doc_right = fitz.open(pdf_right)

    # Render to bitmaps
    pix_left = doc_left[0].get_pixmap(matrix=mat)
    pix_right = doc_right[0].get_pixmap(matrix=mat)

    # Convert to PIL images
    left_img = Image.frombytes("RGB", (pix_left.width, pix_left.height), pix_left.samples)
    right_img = Image.frombytes("RGB", (pix_right.width, pix_right.height), pix_right.samples)

    lw, lh = left_img.size
    rw, rh = right_img.size

    line_width_px = max(1, int(round(line_width_pts * dpi / 72.0)))

    out_w = lw + line_width_px + rw
    out_h = max(lh, rh)

    out_img = Image.new("RGB", (out_w, out_h), "white")

    # Center vertically
    left_y = (out_h - lh) // 2
    right_y = (out_h - rh) // 2

    out_img.paste(left_img, (0, left_y))

    draw = ImageDraw.Draw(out_img)
    x_sep = lw
    draw.rectangle([x_sep, 0, x_sep + line_width_px - 1, out_h], fill="black")

    out_img.paste(right_img, (lw + line_width_px, right_y))

    out_img.save(output_pdf, "PDF", resolution=dpi)
    print(f"✓ Saved combined PDF as {output_pdf}")


combine_pdfs_side_by_side(
    "Figure2ad_final.pdf",
    "Figure2eh_final.pdf",
    "Figure2_final.pdf",
    dpi=300,
    line_width_pts=2
)