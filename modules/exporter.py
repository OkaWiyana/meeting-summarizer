"""
modules/exporter.py
===================
Modul expor hasil ringkasan ke format file:
  - PDF  (menggunakan fpdf2)
  - TXT  (plain text)
"""

from pathlib import Path
from datetime import datetime

from fpdf import FPDF


class _PDFRingkasan(FPDF):
    """Subclass FPDF dengan header dan footer kustom untuk laporan ringkasan."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "LAPORAN RINGKASAN RAPAT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", size=9)
        self.cell(
            0, 6,
            f"Dibuat: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            align="C",
            new_x="LMARGIN", new_y="NEXT",
        )
        self.ln(2)
        self.set_draw_color(100, 100, 100)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Halaman {self.page_no()}", align="C")


def export_to_pdf(
    ringkasan: str,
    transkripsi: str,
    output_path: Path,
) -> Path:
    """
    Ekspor ringkasan dan transkripsi ke file PDF.

    Parameter:
        ringkasan    : Teks ringkasan hasil model.
        transkripsi  : Teks transkripsi yang sudah dibersihkan.
        output_path  : Path lengkap file PDF output.

    Return:
        Path ke file PDF yang berhasil dibuat.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = _PDFRingkasan()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()

    # ── Bagian Ringkasan ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 240, 255)
    pdf.cell(0, 8, "RINGKASAN", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, ringkasan or "(Tidak ada ringkasan)")
    pdf.ln(6)

    # ── Garis pemisah ────────────────────────────────────────
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    # ── Bagian Transkripsi ───────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, "TRANSKRIPSI LENGKAP", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, transkripsi or "(Tidak ada transkripsi)")

    pdf.output(str(output_path))
    return output_path


def export_to_txt(
    ringkasan: str,
    transkripsi: str,
    output_path: Path,
) -> Path:
    """
    Ekspor ringkasan dan transkripsi ke file plain text (.txt).

    Parameter:
        ringkasan    : Teks ringkasan hasil model.
        transkripsi  : Teks transkripsi yang sudah dibersihkan.
        output_path  : Path lengkap file TXT output.

    Return:
        Path ke file TXT yang berhasil dibuat.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")
    konten = (
        f"LAPORAN RINGKASAN RAPAT\n"
        f"Dibuat: {timestamp}\n"
        f"{'=' * 60}\n\n"
        f"RINGKASAN\n"
        f"{'-' * 40}\n"
        f"{ringkasan or '(Tidak ada ringkasan)'}\n\n"
        f"{'=' * 60}\n\n"
        f"TRANSKRIPSI LENGKAP\n"
        f"{'-' * 40}\n"
        f"{transkripsi or '(Tidak ada transkripsi)'}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(konten)

    return output_path
