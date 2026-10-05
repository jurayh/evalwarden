"""In-memory reader for Inspect AI ``.eval`` archives.

An ``.eval`` file is a ZIP archive whose members are JSON documents. This
module returns member bytes without ever extracting an archive path to
disk. ZIP entries written by current Inspect AI use Zstandard (method 93).
Python 3.14's :mod:`zipfile` handles those natively; on older Pythons this
module falls back to the system Zstandard shared library through
:mod:`ctypes` (part of the standard library). If neither decoder is
available, the archive is rejected with a clear :class:`AuditError`
instead of a partial read.
"""
from __future__ import annotations

import binascii
import ctypes
import ctypes.util
import struct
import zipfile
from pathlib import Path
from typing import Any

from . import AuditError

_ZIP_ZSTANDARD = 93
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_OUT_CHUNK = 256 * 1024


class _ZstdInBuffer(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class _ZstdOutBuffer(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


def _zstd_error(lib: Any, code: int) -> str:
    try:
        message = lib.ZSTD_getErrorName(code)
    except Exception:
        return f"Zstandard error code {code}"
    if isinstance(message, bytes):
        return message.decode("utf-8", errors="replace")
    return str(message)


def _load_libzstd() -> Any | None:
    """Load the system Zstandard library, or None when it is unavailable."""
    candidates: list[str] = []
    found = ctypes.util.find_library("zstd")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "libzstd.so.1",
            "libzstd.so",
            "libzstd.dylib",
            "/opt/homebrew/lib/libzstd.dylib",
            "/usr/local/lib/libzstd.dylib",
            "libzstd.dll",
            "zstd.dll",
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            lib = ctypes.CDLL(candidate)
        except OSError:
            continue
        try:
            lib.ZSTD_createDStream.restype = ctypes.c_void_p
            lib.ZSTD_createDStream.argtypes = []
            lib.ZSTD_freeDStream.restype = ctypes.c_size_t
            lib.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
            lib.ZSTD_decompressStream.restype = ctypes.c_size_t
            lib.ZSTD_decompressStream.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ZstdOutBuffer),
                ctypes.POINTER(_ZstdInBuffer),
            ]
            lib.ZSTD_isError.restype = ctypes.c_uint
            lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
            lib.ZSTD_getErrorName.restype = ctypes.c_char_p
            lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        except AttributeError:
            continue
        return lib
    return None


def _decompress_zstd_libzstd(data: bytes, *, member: str) -> bytes:
    """Decompress one ZIP member with the system Zstandard library."""
    lib = _load_libzstd()
    if lib is None:
        raise AuditError(
            f"{member}: this Inspect .eval member uses Zstandard compression, "
            "but no Zstandard decoder is available in this Python/platform "
            "(Python 3.14+ provides one in the standard library; older "
            "Pythons need the system libzstd)"
        )
    stream = lib.ZSTD_createDStream()
    if not stream:
        raise AuditError(f"{member}: could not create a Zstandard decoder")
    try:
        source = ctypes.create_string_buffer(data, len(data))
        in_buffer = _ZstdInBuffer(
            ctypes.cast(source, ctypes.c_void_p), len(data), 0
        )
        output = bytearray()
        while in_buffer.pos < in_buffer.size:
            chunk = ctypes.create_string_buffer(_OUT_CHUNK)
            out_buffer = _ZstdOutBuffer(
                ctypes.cast(chunk, ctypes.c_void_p), _OUT_CHUNK, 0
            )
            code = lib.ZSTD_decompressStream(
                stream, ctypes.byref(out_buffer), ctypes.byref(in_buffer)
            )
            if lib.ZSTD_isError(code):
                raise AuditError(
                    f"{member}: Zstandard decompression failed: "
                    f"{_zstd_error(lib, code)}"
                )
            output.extend(chunk.raw[: out_buffer.pos])
            if out_buffer.pos == 0 and code != 0 and in_buffer.pos >= in_buffer.size:
                break
        return bytes(output)
    finally:
        lib.ZSTD_freeDStream(stream)


def _decompress_zstd(data: bytes, *, member: str) -> bytes:
    """Decompress Zstandard bytes with a stdlib decoder when present."""
    try:
        from compression import zstd  # Python 3.14+

        return zstd.decompress(data)
    except Exception:
        return _decompress_zstd_libzstd(data, member=member)


def _raw_member_bytes(path: Path, info: zipfile.ZipInfo) -> bytes:
    """Read a member's still-compressed bytes using its central-directory
    offset. Only sizes and offsets from the central directory are trusted;
    the local header is parsed solely to find where the data starts."""
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(_LOCAL_HEADER.size)
        if len(header) != _LOCAL_HEADER.size:
            raise AuditError(f"{path}: truncated ZIP local header for {info.filename}")
        fields = _LOCAL_HEADER.unpack(header)
        if fields[0] != 0x04034B50:
            raise AuditError(f"{path}: bad ZIP local header for {info.filename}")
        name_len, extra_len = fields[9], fields[10]
        handle.seek(info.header_offset + _LOCAL_HEADER.size + name_len + extra_len)
        data = handle.read(info.compress_size)
    if len(data) != info.compress_size:
        raise AuditError(f"{path}: truncated ZIP member {info.filename}")
    return data


def _read_member(path: Path, archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.compress_type != _ZIP_ZSTANDARD:
        try:
            return archive.read(info)
        except (KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            raise AuditError(f"{path}: cannot read ZIP member {info.filename}: {exc}") from exc
    try:
        # Python 3.14+ reads Zstandard members through zipfile directly.
        return archive.read(info)
    except (NotImplementedError, RuntimeError):
        pass
    data = _decompress_zstd(_raw_member_bytes(path, info), member=f"{path}:{info.filename}")
    if len(data) != info.file_size:
        raise AuditError(
            f"{path}: ZIP member {info.filename} decompressed to {len(data)} "
            f"bytes, expected {info.file_size}"
        )
    if (binascii.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise AuditError(f"{path}: ZIP member {info.filename} failed its CRC check")
    return data


def read_archive_members(path: Path) -> dict[str, bytes]:
    """Read every live member of a ZIP archive into memory.

    Duplicate member names resolve to the last entry, matching ZIP reader
    semantics (a later member supersedes an earlier one). Nothing is
    written to disk.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AuditError(f"{path}: not a readable ZIP archive: {exc}") from exc
    with archive:
        members: dict[str, bytes] = {}
        # NameToInfo already resolves duplicates to the last entry.
        for name, info in archive.NameToInfo.items():
            if name.endswith("/"):
                continue
            members[name] = _read_member(path, archive, info)
        return members
