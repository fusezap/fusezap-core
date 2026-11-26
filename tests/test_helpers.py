import os, stat
import pytest
from fusezap.helpers import atomic_write

def test_atomic_write_replaces_with_a_plain_create_mode(tmp_path):
  p = tmp_path / "entry"
  p.write_bytes(b"old")
  atomic_write(p, b"new")
  atomic_write(tmp_path / "text.json", "{}")
  umask = os.umask(0); os.umask(umask)
  assert p.read_bytes() == b"new" and (tmp_path / "text.json").read_text() == "{}"
  assert stat.S_IMODE(p.stat().st_mode) == 0o666 & ~umask
  assert sorted(f.name for f in tmp_path.iterdir()) == ["entry", "text.json"]

def test_atomic_write_never_writes_through_a_planted_link(tmp_path):
  victim = tmp_path / "victim"; victim.write_bytes(b"keep")
  d = tmp_path / "cache"; d.mkdir()
  for i in range(64): (d / f".entry.{i}.tmp").symlink_to(victim)
  atomic_write(d / "entry", b"lib")
  assert victim.read_bytes() == b"keep" and (d / "entry").read_bytes() == b"lib" and not (d / "entry").is_symlink()

def test_atomic_write_leaves_no_temp_file_on_failure(tmp_path, monkeypatch):
  def fail(src, dst): raise OSError("disk full")
  monkeypatch.setattr(os, "replace", fail)
  with pytest.raises(OSError, match="disk full"): atomic_write(tmp_path / "entry", b"lib")
  assert list(tmp_path.iterdir()) == []
