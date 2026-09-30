"""The raw delivery runner must enter through Hermes' bootstrap."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_raw_delivery_bootstraps_but_reply_waiter_stays_stdlib_only(tmp_path):
    """The Desktop backend can invoke this file directly, outside the managed launcher."""
    repo = tmp_path / "repo"
    tools_dir = repo / "tools"
    tools_dir.mkdir(parents=True)
    entry = tools_dir / "bot_mode_dm.py"
    source = Path(__file__).resolve().parents[2] / "tools" / "bot_mode_dm.py"
    shutil.copyfile(source, entry)

    marker = tmp_path / "bootstrap-imported"
    (repo / "hermes_bootstrap.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['BOOTSTRAP_MARKER']).write_text('loaded', encoding='utf-8')\n",
        encoding="utf-8",
    )

    def _run(*args):
        return subprocess.run(
            [sys.executable, "-S", str(entry), *args],
            cwd=tmp_path,
            env={
                **os.environ,
                "HERMES_HOME": str(tmp_path / "hermes-home"),
                "BOOTSTRAP_MARKER": str(marker),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )

    delivery = _run("--run-delivery", "invalid-mode")
    assert delivery.returncode == 2, delivery.stderr
    assert marker.read_text(encoding="utf-8") == "loaded"

    marker.unlink()
    reply = tmp_path / "reply.json"
    reply.write_text('{"reply":"ready"}', encoding="utf-8")
    waiter = _run("--wait-reply", str(reply), "teammate", "1")
    assert waiter.returncode == 0, waiter.stderr
    assert waiter.stdout == "Reply from teammate:\nready\n"
    assert not marker.exists()
    assert waiter.stderr == ""
