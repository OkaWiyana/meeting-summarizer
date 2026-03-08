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


def _sanitize_for_pdf(text: str) -> str:
    """
    Mengamankan teks dari karakter Unicode yang tidak didukung oleh
    font standar FPDF (Helvetica/Arial) agar tidak crash.
    Karakter yang tidak didukung akan diubah menjadi tanda tanya (?).
    """
    if not text:
        return ""
    # Ganti smart quotes yang sering bikin crash
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    text = text.replace('—', '-').replace('–', '-')
    # Force encode ke latin-1
    return text.encode('latin-1', 'replace').decode('latin-1')


class _PDFRingkasan(FPDF):
    """Subclass FPDF dengan header dan footer kustom untuk laporan ringkasan."""

    def header(self) -> None:
        # Judul Laporan
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, "LAPORAN RINGKASAN RAPAT", align="C", new_x="LMARGIN", new_y="NEXT")
        
        # Tanggal Dibuat
        self.set_font("Helvetica", size=10)
        self.set_text_color(100, 100, 100)
        self.cell(
            0, 6,
            f"Di-generate otomatis pada: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            align="C",
            new_x="LMARGIN", new_y="NEXT",
        )
        self.ln(3)
        
        # Garis pemisah header
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        # Garis pemisah footer
        self.set_draw_color(220, 220, 220)
        self.line(self.l_margin, self.get_y() - 2, self.w - self.r_margin, self.get_y() - 2)
        self.cell(0, 10, f"Halaman {self.page_no()}", align="C")


def export_to_pdf(
    ringkasan: str,
    transkripsi: str,
    output_path: Path,
) -> Path:
    """
    Ekspor ringkasan dan transkripsi ke file PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = _PDFRingkasan()
    # Menambah margin agar dokumen tidak terlalu sesak
    pdf.set_margins(left=25, top=20, right=25)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Bagian Ringkasan ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(40, 40, 40)
    pdf.set_fill_color(235, 245, 255) # Biru korporat sangat muda
    pdf.cell(0, 8, " RINGKASAN UTAMA", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(50, 50, 50)
    # Gunakan sanitasi di sini
    teks_ringkasan = _sanitize_for_pdf(ringkasan or "(Tidak ada ringkasan)")
    pdf.multi_cell(0, 6.5, teks_ringkasan) 
    pdf.ln(8)

    # ── Bagian Transkripsi ───────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(40, 40, 40)
    pdf.set_fill_color(245, 245, 245) # Abu-abu sangat muda
    pdf.cell(0, 8, " TRANSKRIPSI LENGKAP", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(70, 70, 70)
    # Gunakan sanitasi di sini
    teks_transkripsi = _sanitize_for_pdf(transkripsi or "(Tidak ada transkripsi)")
    pdf.multi_cell(0, 5.5, teks_transkripsi)

    pdf.output(str(output_path))
    return output_path


def export_to_txt(
    ringkasan: str,
    transkripsi: str,
    output_path: Path,
) -> Path:
    """
    Ekspor ringkasan dan transkripsi ke file plain text (.txt).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")
    konten = (
        f"LAPORAN RINGKASAN RAPAT\n"
        f"Di-generate otomatis pada: {timestamp}\n"
        f"{'=' * 65}\n\n"
        f"[ RINGKASAN UTAMA ]\n"
        f"{'-' * 65}\n"
        f"{ringkasan or '(Tidak ada ringkasan)'}\n\n\n"
        f"{'=' * 65}\n\n"
        f"[ TRANSKRIPSI LENGKAP ]\n"
        f"{'-' * 65}\n"
        f"{transkripsi or '(Tidak ada transkripsi)'}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(konten)

    return output_path