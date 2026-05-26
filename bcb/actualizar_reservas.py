#!/usr/bin/env python3

import argparse
import shutil
import re
import unicodedata
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


BCB_URL = "https://www.bcb.gob.bo"
LISTING_QUERY = "estad-sticas-semanales"
REPORT_TYPE = "Información Estadística Semanal"
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reportes"
DATA_DIR = ROOT / "datos"
DEFAULT_EXTRACT_OUTPUT = DATA_DIR / "ultima_reservas.csv"
DEFAULT_SERIES_PATH = DATA_DIR / "reservas.csv"
DEFAULT_BOOTSTRAP_PATH = Path("/home/m/Projects/economia/bcb/reservas.csv")

DATE_REGEX = re.compile(r"(\w+)(?:, )(\d{2})(?: )(\w+)(?:, )(\d{4})")
SPANISH_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
MONTH_NAME_TO_NUMBER = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "adr": 4,
}
TYPE_PATTERNS = [
    ("Divisas", re.compile(r"^divisas")),
    ("DEG", re.compile(r"^deg$")),
    ("Oro", re.compile(r"^oro")),
    ("Posición con el FMI", re.compile(r"^posicion con el fmi")),
    ("Otros", re.compile(r"^otros")),
]
REQUIRED_TYPES = {"Divisas", "DEG", "Oro", "Posición con el FMI"}


@dataclass(frozen=True)
class ReportMetadata:
    publication_date: datetime
    url: str
    extension: str

    @property
    def filename(self) -> str:
        return f"informacion-estadistica-semanal_{self.publication_date:%Y-%m-%d}.{self.extension}"


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value


def parse_listing_date(nodes: Iterable[object]) -> datetime | None:
    for node in nodes:
        match = DATE_REGEX.search(node.get_text())
        if match:
            return datetime(
                year=int(match.group(4)),
                month=SPANISH_MONTHS[match.group(3).lower()],
                day=int(match.group(2)),
            )
    return None


def parse_report_links(nodes: Iterable[object]) -> list[tuple[str, str]]:
    links = []
    for node in nodes:
        href = node["href"]
        extension = href.split(".")[-1].lower()
        if extension in {"xls", "xlsx"}:
            links.append((href, extension))
    return links


def list_reports_page(session: requests.Session, page: int) -> list[ReportMetadata]:
    reports: list[ReportMetadata] = []
    response = session.get(
        BCB_URL,
        params={
            "q": LISTING_QUERY,
            "field_titulo_es_value": REPORT_TYPE,
            "page": str(page),
        },
        timeout=30,
    )
    response.raise_for_status()
    html = BeautifulSoup(response.text, "html.parser")
    items = html.select(".view-content>div")
    for item in items:
        publication_date = parse_listing_date(item.select(".bcb_date"))
        links = parse_report_links(item.select(".bcb_adjunto a"))
        if publication_date and links:
            url, extension = links[0]
            reports.append(
                ReportMetadata(
                    publication_date=publication_date,
                    url=url,
                    extension=extension,
                )
            )
    reports.sort(key=lambda report: report.publication_date, reverse=True)
    return reports


def list_reports_since(session: requests.Session, cutoff_date: datetime | None) -> list[ReportMetadata]:
    reports: list[ReportMetadata] = []
    page = 0
    while True:
        page_reports = list_reports_page(session, page)
        if not page_reports:
            break
        for report in page_reports:
            if cutoff_date is None or report.publication_date.date() > cutoff_date.date():
                reports.append(report)
        if cutoff_date is not None:
            oldest_page_date = min(report.publication_date.date() for report in page_reports)
            if oldest_page_date <= cutoff_date.date():
                break
        page += 1
    reports.sort(key=lambda report: report.publication_date)
    return reports


def resolve_report(session: requests.Session, publication_date: str | None) -> ReportMetadata:
    if publication_date is None:
        page_reports = list_reports_page(session, 0)
        if not page_reports:
            raise RuntimeError("No se encontraron reportes Excel del BCB.")
        return page_reports[0]

    target = datetime.strptime(publication_date, "%Y-%m-%d").date()
    page = 0
    while True:
        page_reports = list_reports_page(session, page)
        if not page_reports:
            break
        for report in page_reports:
            if report.publication_date.date() == target:
                return report
        oldest_page_date = min(report.publication_date.date() for report in page_reports)
        if oldest_page_date < target:
            break
        page += 1
    raise RuntimeError(f"No se encontró un reporte para la fecha de publicación {publication_date}.")


def download_report(session: requests.Session, report: ReportMetadata, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / report.filename
    if output_path.exists():
        return output_path
    response = session.get(report.url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def parse_header_date(value: object) -> datetime | None:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None

    normalized = normalize_text(value)
    if not normalized:
        return None

    match = re.search(
        r"(?P<year>\d{4}).*?(?P<month>ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|adr)",
        normalized,
    )
    if not match:
        return None

    year = int(match.group("year"))
    month = MONTH_NAME_TO_NUMBER[match.group("month")]
    day = monthrange(year, month)[1]
    return datetime(year, month, day)


def resolve_column_dates(df: pd.DataFrame, label_col: int) -> dict[int, datetime]:
    dates: dict[int, datetime] = {}
    for col in range(label_col + 1, df.shape[1]):
        weekly_date = parse_header_date(df.iat[3, col]) if df.shape[0] > 3 else None
        monthly_date = parse_header_date(df.iat[2, col]) if df.shape[0] > 2 else None
        date = weekly_date or monthly_date
        if date is not None:
            dates[col] = date
    if not dates:
        raise RuntimeError("No se pudieron identificar columnas con fecha en el reporte.")
    return dates


def find_section_row(df: pd.DataFrame, pattern: str) -> tuple[int, int]:
    matcher = re.compile(pattern)
    for row_idx in range(df.shape[0]):
        for col_idx in range(df.shape[1]):
            text = normalize_text(df.iat[row_idx, col_idx])
            if text and matcher.search(text):
                return row_idx, col_idx
    raise RuntimeError(f"No se encontró la sección requerida: {pattern}")


def find_type_rows(df: pd.DataFrame, start_row: int, label_col: int) -> dict[str, int]:
    rows: dict[str, int] = {}
    for row_idx in range(start_row + 1, min(start_row + 12, df.shape[0])):
        raw_value = df.iat[row_idx, label_col]
        text = normalize_text(raw_value)
        if not text:
            continue
        for canonical_name, pattern in TYPE_PATTERNS:
            if canonical_name not in rows and pattern.search(text):
                rows[canonical_name] = row_idx
                break
    missing = [name for name in REQUIRED_TYPES if name not in rows]
    if missing:
        raise RuntimeError(
            "No se pudieron encontrar todas las filas de reservas requeridas. "
            f"Faltan: {', '.join(missing)}."
        )
    return rows


def extract_reserves(report_path: Path) -> pd.DataFrame:
    df = pd.read_excel(report_path, sheet_name=0, header=None)
    _, category_col = find_section_row(df, r"operaciones con el exterior")
    reserves_row, reserves_col = find_section_row(df, r"reservas internacionales brutas del bcb")
    label_col = max(category_col, reserves_col)
    column_dates = resolve_column_dates(df, label_col)
    type_rows = find_type_rows(df, reserves_row, label_col)

    records = []
    for reserve_type, row_idx in type_rows.items():
        for col_idx, date in column_dates.items():
            value = pd.to_numeric(pd.Series([df.iat[row_idx, col_idx]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            records.append(
                {
                    "tipo": reserve_type,
                    "fecha": date.date().isoformat(),
                    "valor": float(value),
                }
            )

    if not records:
        raise RuntimeError("La extracción no produjo ninguna fila de reservas.")

    output = pd.DataFrame.from_records(records)
    output = output.drop_duplicates(subset=["tipo", "fecha"], keep="last")
    output = output.sort_values(["fecha", "tipo"]).reset_index(drop=True)
    return output


def load_series(path: Path) -> pd.DataFrame:
    series = pd.read_csv(path)
    expected_columns = ["tipo", "fecha", "valor"]
    if series.columns.tolist() != expected_columns:
        raise RuntimeError(f"{path} no tiene las columnas esperadas: {expected_columns}")
    series["fecha"] = pd.to_datetime(series["fecha"])
    series["valor"] = pd.to_numeric(series["valor"], errors="coerce")
    series = series.dropna(subset=["fecha", "valor"])
    series["fecha"] = series["fecha"].dt.strftime("%Y-%m-%d")
    series = series.drop_duplicates(subset=["tipo", "fecha"], keep="last")
    series = series.sort_values(["fecha", "tipo"]).reset_index(drop=True)
    return series


def bootstrap_series(series_path: Path, bootstrap_path: Path | None) -> None:
    if series_path.exists():
        return
    if bootstrap_path is None or not bootstrap_path.exists():
        raise RuntimeError(
            f"No existe {series_path} y no se encontró un bootstrap válido en {bootstrap_path}."
        )
    series_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bootstrap_path, series_path)


def update_series(
    session: requests.Session,
    series_path: Path,
    reports_dir: Path,
    bootstrap_path: Path | None,
) -> tuple[pd.DataFrame, list[ReportMetadata], int]:
    bootstrap_series(series_path, bootstrap_path)
    series = load_series(series_path)
    last_date = pd.to_datetime(series["fecha"]).max().to_pydatetime()
    reports = list_reports_since(session, last_date)

    added_rows = 0
    processed_reports: list[ReportMetadata] = []
    for report in reports:
        report_path = download_report(session, report, reports_dir)
        extracted = extract_reserves(report_path)
        new_rows = extracted[pd.to_datetime(extracted["fecha"]) > last_date].copy()
        if new_rows.empty:
            continue
        series = pd.concat([series, new_rows], ignore_index=True)
        series = series.drop_duplicates(subset=["tipo", "fecha"], keep="last")
        series = series.sort_values(["fecha", "tipo"]).reset_index(drop=True)
        last_date = pd.to_datetime(series["fecha"]).max().to_pydatetime()
        added_rows += len(new_rows)
        processed_reports.append(report)

    return series, processed_reports, added_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Descarga reportes semanales del BCB y mantiene una serie plana de reservas."
    )
    parser.add_argument(
        "--mode",
        choices=["update", "extract"],
        default="update",
        help="Modo de operación: update para mantener reservas.csv, extract para procesar un solo reporte.",
    )
    parser.add_argument(
        "--publication-date",
        help="Fecha de publicación del reporte en formato YYYY-MM-DD. En modo extract, por defecto usa el último disponible.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help=f"Directorio local para guardar reportes descargados. Por defecto: {REPORTS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACT_OUTPUT,
        help=f"Ruta del CSV de salida en modo extract. Por defecto: {DEFAULT_EXTRACT_OUTPUT}",
    )
    parser.add_argument(
        "--series-path",
        type=Path,
        default=DEFAULT_SERIES_PATH,
        help=f"Ruta del CSV maestro en modo update. Por defecto: {DEFAULT_SERIES_PATH}",
    )
    parser.add_argument(
        "--bootstrap-from",
        type=Path,
        default=DEFAULT_BOOTSTRAP_PATH,
        help=(
            "CSV base para inicializar la serie si no existe. "
            f"Por defecto: {DEFAULT_BOOTSTRAP_PATH}"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = requests.Session()
    if args.mode == "extract":
        report = resolve_report(session, args.publication_date)
        report_path = download_report(session, report, args.reports_dir)
        reserves = extract_reserves(report_path)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        reserves.to_csv(args.output, index=False)

        print(f"Modo: extract")
        print(f"Reporte: {report.publication_date:%Y-%m-%d}")
        print(f"Archivo local: {report_path}")
        print(f"Filas extraídas: {len(reserves)}")
        print(f"Fechas: {reserves['fecha'].min()} -> {reserves['fecha'].max()}")
        print(f"CSV: {args.output}")
        return

    series, processed_reports, added_rows = update_series(
        session=session,
        series_path=args.series_path,
        reports_dir=args.reports_dir,
        bootstrap_path=args.bootstrap_from,
    )
    args.series_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.series_path, index=False)

    print("Modo: update")
    print(f"CSV maestro: {args.series_path}")
    print(f"Filas totales: {len(series)}")
    print(f"Fechas: {series['fecha'].min()} -> {series['fecha'].max()}")
    print(f"Reportes procesados con datos nuevos: {len(processed_reports)}")
    print(f"Filas agregadas: {added_rows}")
    if processed_reports:
        print(
            "Publicaciones usadas: "
            + ", ".join(report.publication_date.strftime("%Y-%m-%d") for report in processed_reports)
        )
    else:
        print("Sin cambios: no se encontraron fechas nuevas para agregar.")


if __name__ == "__main__":
    main()
