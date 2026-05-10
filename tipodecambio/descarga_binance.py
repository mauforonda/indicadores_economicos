#!/usr/bin/env python3

"""Descarga snapshots ZIP de Binance desde Google Drive y los consolida a parquet.

Supuestos actuales:
1. Existe un remote de `rclone` que apunta directamente al folder de Google Drive.
2. Existe un manifiesto local con una lista de archivos `raw-data_*.zip`, uno por linea.
3. Cada ZIP contiene `data/raw-data.csv`.

Configuracion sugerida del remote:
1. Ejecuta `rclone config`.
2. Crea un remote nuevo, por ejemplo `gdrive_binance`, de tipo `drive`.
3. Autoriza con la cuenta que tiene acceso al folder.
4. En la configuracion avanzada, fija `root_folder_id` al folder compartido.
5. Verifica acceso con `rclone lsf gdrive_binance: --files-only --include "*.zip"`.

Comando para reproducir o actualizar el manifiesto:
`rclone lsf gdrive_binance: --files-only --include "*.zip" --format "p" --timeout 60s > tipodecambio/datos/binance_historico_muestras`

El flujo del script es:
1. Lee el manifiesto local de nombres ZIP.
2. Filtra por el timestamp UTC embebido en el filename, pero usando rango en hora Bolivia.
3. Incluye el ultimo snapshot previo al inicio para preservar `tranzado_estimado`.
4. Descarga solo esos ZIP a un staging temporal.
5. Extrae `data/raw-data.csv` de cada ZIP a un directorio persistente de CSV.
6. Consolida las columnas necesarias a un parquet en `tipodecambio/datos/`.

Si un ZIP falla al descargar o no contiene `data/raw-data.csv`, se reporta en stderr y el
script continua con los demas archivos.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ZONA_HORARIA_BOLIVIA = "America/La_Paz"
PATRON_ARCHIVO_ZIP = re.compile(r"^raw-data_(\d{14})\.zip$")
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
RUTA_MANIFIESTO_POR_DEFECTO = DIRECTORIO_SALIDA / "binance_historico_muestras"
RUTA_INTERNA_CSV = "data/raw-data.csv"


@dataclass(frozen=True)
class ArchivoRemoto:
    """Representa un snapshot remoto identificado por su filename ZIP."""

    nombre_zip: str
    timestamp_utc: pd.Timestamp


def imprimir_estado(mensaje: str) -> None:
    """Emite mensajes operativos a stderr."""
    print(mensaje, file=sys.stderr)


def anexar_log(ruta: Path | None, mensaje: str) -> None:
    """Anexa una linea con timestamp a un logfile opcional."""
    if ruta is None:
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().isoformat(timespec="seconds")
    with ruta.open("a", encoding="utf-8") as handle:
        handle.write(f"[{marca}] {mensaje}\n")


def parsear_fecha_local(texto: str, *, es_fin: bool) -> pd.Timestamp:
    """Parsea una fecha/hora en zona Bolivia.

    Si el texto solo trae fecha, se interpreta como dia completo:
    - inicio: 00:00:00
    - fin: 23:59:59.999999
    """

    base = pd.Timestamp(texto)
    if base.tzinfo is None:
        marca = base.tz_localize(ZONA_HORARIA_BOLIVIA)
    else:
        marca = base.tz_convert(ZONA_HORARIA_BOLIVIA)

    if len(texto) == 10:
        if es_fin:
            return marca + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return marca.normalize()
    return marca


def formatear_etiqueta_archivo(fecha: pd.Timestamp) -> str:
    """Convierte una marca de tiempo local en un fragmento estable para filename."""
    return fecha.tz_convert(ZONA_HORARIA_BOLIVIA).strftime("%Y%m%dT%H%M%S")


def ejecutar(
    comando: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Ejecuta un comando y opcionalmente falla con contexto."""
    try:
        resultado = subprocess.run(
            comando,
            check=False,
            cwd=cwd,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"No se encontro el ejecutable requerido: {comando[0]}") from exc

    if check and resultado.returncode != 0:
        detalle = resultado.stderr.strip() or resultado.stdout.strip()
        raise SystemExit(f"Fallo el comando `{' '.join(comando)}`\n{detalle}")
    return resultado


def ejecutar_streaming(
    comando: list[str],
    *,
    cwd: Path | None = None,
) -> int:
    """Ejecuta un comando heredando stdout/stderr para mostrar progreso real."""
    try:
        proceso = subprocess.run(
            comando,
            check=False,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"No se encontro el ejecutable requerido: {comando[0]}") from exc
    return proceso.returncode


def leer_manifiesto(ruta: Path) -> list[ArchivoRemoto]:
    """Lee el manifiesto local y parsea nombres validos de ZIP."""
    if not ruta.exists():
        raise SystemExit(f"No existe el manifiesto `{ruta}`.")

    snapshots: list[ArchivoRemoto] = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        nombre = linea.strip()
        if not nombre:
            continue
        coincidencia = PATRON_ARCHIVO_ZIP.match(nombre)
        if coincidencia is None:
            continue
        timestamp_utc = pd.to_datetime(
            coincidencia.group(1),
            format="%Y%m%d%H%M%S",
            utc=True,
        )
        snapshots.append(ArchivoRemoto(nombre_zip=nombre, timestamp_utc=timestamp_utc))

    snapshots.sort(key=lambda item: item.timestamp_utc)
    if not snapshots:
        raise SystemExit(f"El manifiesto `{ruta}` no contiene ZIPs validos.")
    return snapshots


def actualizar_manifiesto(remote: str, ruta: Path) -> None:
    """Regenera el manifiesto local consultando el remote de rclone."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    imprimir_estado(f"Actualizando manifiesto en {ruta} desde `{remote}:`...")
    resultado = ejecutar(
        [
            "rclone",
            "lsf",
            f"{remote}:",
            "--files-only",
            "--include",
            "*.zip",
            "--format",
            "p",
            "--timeout",
            "60s",
        ]
    )
    ruta.write_text(resultado.stdout, encoding="utf-8")
    cantidad = sum(1 for linea in resultado.stdout.splitlines() if linea.strip())
    imprimir_estado(f"Manifiesto actualizado con {cantidad} ZIPs.")


def preparar_directorio_csv(ruta: Path) -> None:
    """Crea o limpia el directorio persistente de CSV de la corrida."""
    if ruta.exists():
        shutil.rmtree(ruta)
    ruta.mkdir(parents=True, exist_ok=True)


def seleccionar_archivos(
    archivos: list[ArchivoRemoto],
    inicio_local: pd.Timestamp,
    fin_local: pd.Timestamp,
) -> list[ArchivoRemoto]:
    """Filtra snapshots por rango local e incluye contexto previo al inicio."""
    seleccionados: list[ArchivoRemoto] = []
    contexto_previo: ArchivoRemoto | None = None

    for archivo in archivos:
        marca_local = archivo.timestamp_utc.tz_convert(ZONA_HORARIA_BOLIVIA)
        if marca_local < inicio_local:
            contexto_previo = archivo
            continue
        if marca_local > fin_local:
            break
        seleccionados.append(archivo)

    if contexto_previo is not None:
        seleccionados.insert(0, contexto_previo)

    return seleccionados


def descargar_archivos(
    origen: str,
    archivos: list[ArchivoRemoto],
    destino: Path,
    *,
    verbose: bool,
    log_file: Path | None,
) -> tuple[list[Path], list[str]]:
    """Descarga ZIPs seleccionados y devuelve los disponibles localmente.

    Si `rclone` falla parcialmente, se registran faltantes y el script continua.
    """
    destino.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        lista_temporal = Path(handle.name)
        for archivo in archivos:
            handle.write(f"{archivo.nombre_zip}\n")

    try:
        comando = [
            "rclone",
            "copy",
            origen,
            str(destino),
            "--files-from",
            str(lista_temporal),
            "--no-traverse",
            "--transfers",
            str(min(16, max(4, os.cpu_count() or 4))),
            "--checkers",
            str(min(16, max(4, os.cpu_count() or 4))),
            "--timeout",
            "60s",
            "--progress",
            "--stats",
            "15s",
        ]
        if verbose:
            if log_file is None:
                raise SystemExit("`--verbose` requiere una ruta de log disponible.")
            comando.extend(
                [
                    "--use-json-log",
                    "--log-level",
                    "DEBUG",
                    "--log-file",
                    str(log_file),
                ]
            )
            anexar_log(log_file, f"Iniciando rclone copy desde {origen}")
        retorno = ejecutar_streaming(comando)
    finally:
        lista_temporal.unlink(missing_ok=True)

    if retorno != 0:
        imprimir_estado("Advertencia: `rclone copy` devolvio un codigo no cero.")
        anexar_log(log_file, "Advertencia: `rclone copy` devolvio un codigo no cero.")

    descargados: list[Path] = []
    faltantes: list[str] = []
    for archivo in archivos:
        ruta_local = destino / archivo.nombre_zip
        if ruta_local.exists() and ruta_local.stat().st_size > 0:
            descargados.append(ruta_local)
        else:
            faltantes.append(archivo.nombre_zip)

    return descargados, faltantes


def extraer_csvs(zip_paths: list[Path], destino_csv: Path) -> tuple[list[Path], list[str]]:
    """Extrae `data/raw-data.csv` de cada ZIP a staging temporal."""
    destino_csv.mkdir(parents=True, exist_ok=True)
    extraidos: list[Path] = []
    problemas: list[str] = []
    total = len(zip_paths)

    for indice, zip_path in enumerate(zip_paths, start=1):
        if indice == 1 or indice % 250 == 0 or indice == total:
            imprimir_estado(f"Extrayendo ZIPs: {indice}/{total}")
        nombre_csv = zip_path.name.replace(".zip", ".csv")
        ruta_csv = destino_csv / nombre_csv
        try:
            with zipfile.ZipFile(zip_path) as archivo_zip:
                miembros = set(archivo_zip.namelist())
                if RUTA_INTERNA_CSV not in miembros:
                    problemas.append(f"{zip_path.name}: no contiene `{RUTA_INTERNA_CSV}`")
                    continue
                info = archivo_zip.getinfo(RUTA_INTERNA_CSV)
                if info.file_size == 0:
                    problemas.append(f"{zip_path.name}: `{RUTA_INTERNA_CSV}` esta vacio")
                    continue
                with archivo_zip.open(RUTA_INTERNA_CSV) as origen, ruta_csv.open("wb") as salida:
                    salida.write(origen.read())
            if ruta_csv.stat().st_size == 0:
                problemas.append(f"{zip_path.name}: CSV extraido vacio")
                ruta_csv.unlink(missing_ok=True)
                continue
            extraidos.append(ruta_csv)
        except zipfile.BadZipFile:
            problemas.append(f"{zip_path.name}: ZIP corrupto o invalido")
        except OSError as exc:
            problemas.append(f"{zip_path.name}: error al extraer ({exc})")

    return extraidos, problemas


def consolidar_a_parquet(csv_dir: Path, parquet_salida: Path) -> None:
    """Usa DuckDB CLI para proyectar columnas y escribir un parquet compacto."""
    consulta = f"""
    COPY (
        WITH crudo AS (
            SELECT
                *
            FROM read_csv_auto(
                '{csv_dir.as_posix()}/*.csv',
                union_by_name = true,
                all_varchar = true,
                sample_size = -1
            )
        )
        SELECT
            TRY_CAST("timestamp" AS BIGINT) AS timestamp,
            "adv.asset" AS asset,
            "adv.tradeType" AS tradetype,
            "adv.advNo" AS advno,
            "adv.fiatUnit" AS fiatunit,
            TRY_CAST("adv.price" AS DOUBLE) AS price,
            TRY_CAST("adv.tradableQuantity" AS DOUBLE) AS tradablequantity,
            "advertiser.userNo" AS advertiser_userno,
            TRY_CAST("adv.minSingleTransAmount" AS DOUBLE) AS minsingletransamount,
            TRY_CAST("adv.maxSingleTransAmount" AS DOUBLE) AS maxsingletransamount
        FROM crudo
        WHERE TRY_CAST("timestamp" AS BIGINT) IS NOT NULL
    )
    TO '{parquet_salida.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    ejecutar(["duckdb", "-c", consulta])


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de linea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Descarga ZIPs `raw-data_*.zip` desde un remote de rclone y genera "
            "binance_historico_<inicio>_<fin>.parquet."
        ),
        epilog=(
            "Ejemplo de configuracion:\n"
            "  rclone config\n"
            "  # crear remote `gdrive_binance` tipo drive, autorizado y apuntando al folder\n"
            "  rclone lsf gdrive_binance: --files-only --include '*.zip'\n\n"
            "Para regenerar el manifiesto:\n"
            "  rclone lsf gdrive_binance: --files-only --include '*.zip' --format 'p' --timeout 60s > tipodecambio/datos/binance_historico_muestras\n"
            "  # o usar `--actualizar-manifest`\n\n"
            "Ejemplo de uso:\n"
            "  ./tipodecambio/descarga_binance.py --remote gdrive_binance "
            "--inicio 2025-08-08 --fin 2025-11-08"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--remote",
        required=True,
        help="Nombre del remote de rclone que apunta al folder de Google Drive.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=RUTA_MANIFIESTO_POR_DEFECTO,
        help="Archivo local con la lista de `raw-data_*.zip` disponibles.",
    )
    parser.add_argument(
        "--actualizar-manifest",
        action="store_true",
        help="Regenera el manifiesto local consultando el remote antes de filtrar.",
    )
    parser.add_argument(
        "--inicio",
        required=True,
        help="Inicio del rango en hora Bolivia. Ej: 2025-08-08 o 2025-08-08T00:00:00",
    )
    parser.add_argument(
        "--fin",
        required=True,
        help="Fin del rango en hora Bolivia. Ej: 2025-11-08 o 2025-11-08T23:59:59",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="Ruta opcional del parquet de salida. Por defecto usa tipodecambio/datos/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Escribe logs DEBUG de rclone a un logfile mientras mantiene el progreso en terminal.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Ruta del logfile usado con `--verbose`. Por defecto usa tipodecambio/datos/descarga_binance.log",
    )
    return parser


def main() -> None:
    """Ejecuta la descarga incremental y consolidacion a parquet."""
    args = construir_parser().parse_args()
    inicio_local = parsear_fecha_local(args.inicio, es_fin=False)
    fin_local = parsear_fecha_local(args.fin, es_fin=True)
    if fin_local < inicio_local:
        raise SystemExit("`--fin` no puede ser anterior a `--inicio`.")

    etiqueta_inicio = formatear_etiqueta_archivo(inicio_local)
    etiqueta_fin = formatear_etiqueta_archivo(fin_local)
    ruta_salida = args.salida or (
        DIRECTORIO_SALIDA / f"binance_historico_{etiqueta_inicio}_{etiqueta_fin}.parquet"
    )
    directorio_csv = DIRECTORIO_SALIDA / f"binance_csv_{etiqueta_inicio}_{etiqueta_fin}"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_log = args.log_file.resolve() if args.log_file else DIRECTORIO_SALIDA / "descarga_binance.log"
    if args.verbose:
        imprimir_estado(f"Logs detallados en {ruta_log}")
        anexar_log(ruta_log, "Inicio de ejecucion de descarga_binance.py")

    ruta_manifiesto = args.manifest.resolve()
    if args.actualizar_manifest or not ruta_manifiesto.exists():
        anexar_log(ruta_log if args.verbose else None, f"Actualizando manifiesto desde `{args.remote}:`")
        actualizar_manifiesto(args.remote, ruta_manifiesto)

    archivos = leer_manifiesto(ruta_manifiesto)
    seleccionados = seleccionar_archivos(archivos, inicio_local, fin_local)
    if not seleccionados:
        raise SystemExit("No se encontraron snapshots en el rango solicitado.")

    imprimir_estado(
        f"Seleccionados {len(seleccionados)} ZIPs entre {inicio_local} y {fin_local}."
    )
    anexar_log(
        ruta_log if args.verbose else None,
        f"Seleccionados {len(seleccionados)} ZIPs entre {inicio_local} y {fin_local}.",
    )
    preparar_directorio_csv(directorio_csv)
    imprimir_estado(f"CSV persistentes en {directorio_csv}")
    anexar_log(
        ruta_log if args.verbose else None,
        f"CSV persistentes en {directorio_csv}",
    )

    with tempfile.TemporaryDirectory(prefix="binance_zip_", dir=DIRECTORIO_SALIDA) as temp_zip_dir:
        staging_zip = Path(temp_zip_dir)

        descargados, faltantes = descargar_archivos(
            f"{args.remote}:",
            seleccionados,
            staging_zip,
            verbose=args.verbose,
            log_file=ruta_log if args.verbose else None,
        )
        imprimir_estado(f"Descargados {len(descargados)} ZIPs.")
        anexar_log(ruta_log if args.verbose else None, f"Descargados {len(descargados)} ZIPs.")
        if faltantes:
            imprimir_estado(f"Advertencia: faltaron {len(faltantes)} ZIPs durante la descarga.")
            for nombre in faltantes[:20]:
                imprimir_estado(f"  - {nombre}")
            if len(faltantes) > 20:
                imprimir_estado(f"  ... y {len(faltantes) - 20} mas")
            anexar_log(
                ruta_log if args.verbose else None,
                f"Advertencia: faltaron {len(faltantes)} ZIPs durante la descarga.",
            )

        if not descargados:
            raise SystemExit("No se pudo descargar ningun ZIP util.")

        extraidos, problemas = extraer_csvs(descargados, directorio_csv)
        imprimir_estado(f"Extraidos {len(extraidos)} CSVs validos.")
        anexar_log(ruta_log if args.verbose else None, f"Extraidos {len(extraidos)} CSVs validos.")
        if problemas:
            imprimir_estado(f"Advertencia: hubo {len(problemas)} problemas al extraer ZIPs.")
            for detalle in problemas[:20]:
                imprimir_estado(f"  - {detalle}")
            if len(problemas) > 20:
                imprimir_estado(f"  ... y {len(problemas) - 20} mas")
            anexar_log(
                ruta_log if args.verbose else None,
                f"Advertencia: hubo {len(problemas)} problemas al extraer ZIPs.",
            )

        if not extraidos:
            raise SystemExit("No se pudo extraer ningun CSV valido de los ZIPs descargados.")

        anexar_log(ruta_log if args.verbose else None, f"Consolidando parquet en {ruta_salida}")
        consolidar_a_parquet(directorio_csv, ruta_salida)

    anexar_log(ruta_log if args.verbose else None, f"Parquet generado en {ruta_salida}")
    print(ruta_salida)


if __name__ == "__main__":
    main()
