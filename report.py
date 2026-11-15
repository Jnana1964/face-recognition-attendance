"""Build an Excel attendance register: students x dates, P/A, total and percentage.

    python report.py        ->  attendance_report.xlsx
"""
import csv
import glob
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as L

from common import ATTENDANCE_DIR, BASE, STUDENTS_CSV

REPORT_PATH = os.path.join(BASE, "attendance_report.xlsx")


def load_register():
    """Return (students, days) where days = {date: set(ids present)}."""
    students = []
    if os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, newline="", encoding="utf-8") as f:
            students = list(csv.DictReader(f))
    days = {}
    for path in sorted(glob.glob(os.path.join(ATTENDANCE_DIR, "*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            days[os.path.splitext(os.path.basename(path))[0]] = {r["id"] for r in csv.DictReader(f)}
    return students, days


def build_report(out=REPORT_PATH):
    students, days = load_register()
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    ws.append(["Reg. No", "Name"] + list(days) + ["Present", "Total", "%"])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(horizontal="center")

    green, red = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="FFC7CE")
    for r, s in enumerate(students, start=2):
        ws.append([s["id"], s["name"]] + ["P" if s["id"] in p else "A" for p in days.values()])
        first, last = 3, 2 + len(days)
        for col in range(first, last + 1):
            cell = ws.cell(row=r, column=col)
            cell.fill = green if cell.value == "P" else red
            cell.alignment = Alignment(horizontal="center")
        if days:
            rng = f"{L(first)}{r}:{L(last)}{r}"
            ws.cell(row=r, column=last + 1, value=f'=COUNTIF({rng},"P")')
            ws.cell(row=r, column=last + 2, value=len(days))
            ws.cell(row=r, column=last + 3, value=f"=ROUND({L(last + 1)}{r}/{L(last + 2)}{r}*100,1)")

    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 24
    wb.save(out)
    return out, len(students), len(days)


if __name__ == "__main__":
    path, n_students, n_days = build_report()
    print(f"Saved {path}  ({n_students} students, {n_days} days)")
