"""Export functions for crack image analysis results."""

import os
import io
import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak,
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def _log(log_callback, msg):

    if log_callback:
        log_callback(msg)
    else:
        print(msg)


def _save_image_to_file(file_path, img):

    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".png"
        is_success, im_buf = cv2.imencode(ext, img)
        if is_success:
            im_buf.tofile(file_path)
            return True
    except Exception as e:
        print(f"Error saving image {file_path}: {e}")
    return False



def export_excel(analysis_data, segments_data, file_path, log_callback=None):
    """
    Export analysis data to an Excel (.xlsx) file.

    Parameters
    ----------
    analysis_data : dict
        Overall analysis statistics (key -> value).
    segments_data : list[dict]
        Per-segment crack details.
    file_path : str
        Destination .xlsx path.
    log_callback : callable(str) or None
        If given, called with status/diagnostic messages.
    """
    if not file_path:
        _log(log_callback, "Warning: No file path provided for Excel export")
        return

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_summary = pd.DataFrame([analysis_data])
            df_summary.to_excel(writer, sheet_name='Overall Statistics', index=False)
            if segments_data:
                df_segments = pd.DataFrame(segments_data)
                df_segments.to_excel(writer, sheet_name='Crack Segment Details', index=False)
        _log(log_callback, f"Export succeeded: {file_path}")
    except Exception as e:
        _log(log_callback, f"Export failed: {e}")
        raise



def export_csv(analysis_data, segments_data, file_path, log_callback=None):
    """
    Export analysis data to CSV.  Segment details go to a companion file
    named ``<root>_segments.csv`` when segment data is present.

    Parameters
    ----------
    analysis_data : dict
        Overall analysis statistics (key -> value).
    segments_data : list[dict]
        Per-segment crack details.
    file_path : str
        Destination .csv path.
    log_callback : callable(str) or None
        If given, called with status/diagnostic messages.
    """
    if not file_path:
        _log(log_callback, "Warning: No file path provided for CSV export")
        return

    try:
        df_summary = pd.DataFrame([analysis_data])
        df_summary.to_csv(file_path, index=False, encoding='utf-8-sig')
        if segments_data:
            root, ext = os.path.splitext(file_path)
            segments_path = f"{root}_segments{ext or '.csv'}"
            df_segments = pd.DataFrame(segments_data)
            df_segments.to_csv(segments_path, index=False, encoding='utf-8-sig')
            _log(log_callback, f"Export succeeded: {file_path} and {segments_path}")
        else:
            _log(log_callback, f"Export succeeded: {file_path}")
    except Exception as e:
        _log(log_callback, f"Export failed: {e}")
        raise


# 3. PDF report export

def export_pdf(analysis_data, segments_data, cv_image, binary_image,
               final_image, file_path, log_callback=None):
    """
    Export a full PDF analysis report containing statistics tables and
    embedded source / binary / result images.

    Parameters
    ----------
    analysis_data : dict
        Overall analysis statistics (key -> value).
    segments_data : list[dict]
        Per-segment crack details.
    cv_image : np.ndarray or None
        Original/source image (BGR or grayscale).
    binary_image : np.ndarray or None
        Binarized image.
    final_image : np.ndarray or None
        Result/preview image.
    file_path : str
        Destination .pdf path.
    log_callback : callable(str) or None
        If given, called with status/diagnostic messages.
    """
    if final_image is None:
        _log(log_callback, "Warning: No data to export")
        return

    if not file_path:
        _log(log_callback, "Warning: No file path provided for PDF export")
        return

    try:
        # ---- Chinese font detection ----
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        font_name = "ChineseFont"
        font_registered = False

        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    pdfmetrics.registerFont(TTFont(font_name, fp))
                    font_registered = True
                    break
                except Exception:
                    continue

        doc = SimpleDocTemplate(
            file_path, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()

        if font_registered:
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=18,
                spaceAfter=30,
                alignment=1,
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
            )
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=10,
            )
            heading3_style = ParagraphStyle(
                'CustomHeading3',
                parent=styles['Heading3'],
                fontName=font_name,
                fontSize=12,
            )
        else:
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1,
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=20,
            )
            normal_style = styles['Normal']
            heading3_style = styles['Heading3']

        story = []

        # Title
        story.append(Paragraph("Crack Analysis Report", title_style))
        story.append(Paragraph(
            f"Generated Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        ))
        story.append(Spacer(1, 20))

        # ---- 1. Overall Statistics ----
        story.append(Paragraph("1. Overall Statistics", heading_style))

        data_table = [["Parameter", "Value"]]
        for key, value in analysis_data.items():
            if isinstance(value, float):
                data_table.append([key, f"{value:.4f}"])
            else:
                data_table.append([key, str(value)])

        table = Table(data_table, colWidths=[8 * cm, 8 * cm])
        table_style_list = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]
        if font_registered:
            table_style_list.append(('FONTNAME', (0, 0), (-1, -1), font_name))
        table.setStyle(TableStyle(table_style_list))
        story.append(table)
        story.append(Spacer(1, 20))

        # ---- 2. Analysis Images ----
        story.append(Paragraph("2. Analysis Images", heading_style))

        images_to_add = []
        img_labels = ["Source Image", "Binarized Image", "Result Preview"]
        img_sources = [cv_image, binary_image, final_image]

        for label, img in zip(img_labels, img_sources):
            if img is None:
                continue
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            max_width = 15 * cm
            max_height = 10 * cm
            pil_img.thumbnail(
                (max_width * 2, max_height * 2), Image.Resampling.LANCZOS,
            )

            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            img_width = min(pil_img.width / 2, max_width)
            img_height = min(pil_img.height / 2, max_height)

            rl_img = RLImage(img_buffer, width=img_width, height=img_height)
            images_to_add.append((label, rl_img))

        for i, (label, rl_img) in enumerate(images_to_add):
            story.append(Paragraph(f"2.{i+1} {label}", heading3_style))
            story.append(rl_img)
            story.append(Spacer(1, 15))

        # ---- 3. Crack Segment Details ----
        if segments_data:
            story.append(PageBreak())
            story.append(Paragraph("3. Crack Segment Details", heading_style))

            seg_table = [["No."] + list(segments_data[0].keys())]
            for idx, seg in enumerate(segments_data, 1):
                row = [str(idx)]
                for key, value in seg.items():
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                seg_table.append(row)

            col_width = 16 * cm / len(seg_table[0])
            seg_tbl = Table(seg_table, colWidths=[col_width] * len(seg_table[0]))
            seg_style_list = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]
            if font_registered:
                seg_style_list.append(('FONTNAME', (0, 0), (-1, -1), font_name))
            seg_tbl.setStyle(TableStyle(seg_style_list))
            story.append(seg_tbl)

        doc.build(story)
        _log(log_callback, f"PDF export succeeded: {file_path}")

    except Exception as e:
        _log(log_callback, f"PDF export failed: {e}")
        raise



def export_binary_image(final_image, file_path, is_circular_mode=False,
                        circular_mask=None, log_callback=None):
    """Export the binarized crack image to a PNG file."""
    if final_image is None:
        _log(log_callback, "Warning: No data to export")
        return

    if not file_path:
        _log(log_callback, "Warning: No file path provided for binary image export")
        return

    try:
        # Create a white canvas matching the final image dimensions.
        if len(final_image.shape) == 2:
            h, w = final_image.shape
            export_img = np.ones((h, w), dtype=np.uint8) * 255
        else:
            h, w = final_image.shape[:2]
            export_img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Invert the binary so cracks become black.
        binary = cv2.bitwise_not(final_image)

        if is_circular_mode and circular_mask is not None:
            # Inside the circular mask we use the inverted binary;
            # outside remains white (the default canvas).
            export_img[circular_mask > 0] = binary[circular_mask > 0]
        else:
            # Rectangle mode: directly export the binarized result.
            export_img = binary

        if not _save_image_to_file(file_path, export_img):
            raise OSError(f"Failed to save image: {file_path}")

        _log(log_callback, f"Binary image export succeeded: {file_path}")

    except Exception as e:
        _log(log_callback, f"Export failed: {e}")
        raise
