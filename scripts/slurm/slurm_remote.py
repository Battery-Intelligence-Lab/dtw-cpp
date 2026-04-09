"""Generalized SSH/SFTP remote helper for SLURM HPC clusters.

Reads cluster config from .env at project root. SSH-key-first auth with
paramiko. Two-hop gateway support. Explicit upload allowlist.

Usage:
    uv run scripts/slurm/slurm_remote.py test                       # Test connection
    uv run scripts/slurm/slurm_remote.py setup-keys                 # Copy SSH key
    uv run scripts/slurm/slurm_remote.py upload                     # Upload code + data
    uv run scripts/slurm/slurm_remote.py build --profile h100       # Build on cluster
    uv run scripts/slurm/slurm_remote.py submit-cpu                 # Submit CPU test
    uv run scripts/slurm/slurm_remote.py submit-gpu                 # Submit GPU test
    uv run scripts/slurm/slurm_remote.py submit-checkpoint          # Submit checkpoint test
    uv run scripts/slurm/slurm_remote.py submit-parquet             # Submit Parquet test
    uv run scripts/slurm/slurm_remote.py status                     # SLURM queue
    uv run scripts/slurm/slurm_remote.py stats --job 12345          # sacct stats
    uv run scripts/slurm/slurm_remote.py download                   # Download results
    uv run scripts/slurm/slurm_remote.py ssh "hostname"             # Run command
    uv run scripts/slurm/slurm_remote.py interactive                # Session guide
    uv run scripts/slurm/slurm_remote.py --trust-unknown-hosts test # Accept unknown keys

Config: copy scripts/slurm/env.example to .env at project root and edit.
"""
from __future__ import annotations
import argparse, getpass, os, sys, time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional
import paramiko

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Upload allowlist -- NEVER upload .env, .git/, build*/, arc/
UPLOAD_DIRS = ["dtwc", "cmake", "scripts/slurm"]
UPLOAD_FILES = ["CMakeLists.txt", "VERSION", "CMakePresets.json"]
UPLOAD_DATA = [
    "data/benchmark/UCRArchive_2018/Coffee",
    "data/benchmark/UCRArchive_2018/Beef",
]


@dataclass
class Config:
    user: str = ""; host: str = ""; gateway: str = ""
    password: Optional[str] = None; ssh_key: str = ""
    home_folder: str = ""; data_folder: str = ""; remote_base: str = ""
    local_data: str = "data/benchmark/UCRArchive_2018"
    partition: str = "short"; cluster: str = ""
    gpu_gres: str = "gpu:1"; email: str = ""
    trust_unknown_hosts: bool = False


def _load_env_file() -> dict[str, str]:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip()
    return result


def _env(env: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key, env.get(key, default))


def load_config(trust_unknown_hosts: bool = False) -> Config:
    env = _load_env_file()
    cfg = Config(
        user=_env(env, "SLURM_USER"), host=_env(env, "SLURM_HOST"),
        gateway=_env(env, "SLURM_GATEWAY"), ssh_key=_env(env, "SLURM_SSH_KEY"),
        home_folder=_env(env, "SLURM_HOME_FOLDER"),
        data_folder=_env(env, "SLURM_DATA_FOLDER"),
        remote_base=_env(env, "SLURM_REMOTE_BASE"),
        local_data=_env(env, "SLURM_LOCAL_DATA", "data/benchmark/UCRArchive_2018"),
        partition=_env(env, "SLURM_PARTITION", "short"),
        cluster=_env(env, "SLURM_CLUSTER"),
        gpu_gres=_env(env, "SLURM_GPU_GRES", "gpu:1"),
        email=_env(env, "SLURM_EMAIL"), trust_unknown_hosts=trust_unknown_hosts,
    )
    if not cfg.user or not cfg.host:
        print("ERROR: SLURM_USER and SLURM_HOST required. Copy scripts/slurm/env.example to .env")
        sys.exit(1)
    if not cfg.remote_base:
        base = cfg.data_folder or cfg.home_folder
        if not base:
            print("ERROR: Set SLURM_REMOTE_BASE, SLURM_DATA_FOLDER, or SLURM_HOME_FOLDER.")
            sys.exit(1)
        cfg.remote_base = f"{base}/dtw-cpp"
    return cfg


def _load_password() -> Optional[str]:
    pw = os.environ.get("SLURM_PASSWORD")
    if pw:
        return pw
    pw = _load_env_file().get("SLURM_PASSWORD", "")
    return pw if pw and pw != "CHANGE_ME" else None


def _ssh_key_path(cfg: Config) -> Optional[Path]:
    if cfg.ssh_key:
        p = Path(os.path.expanduser(cfg.ssh_key))
        if p.exists():
            return p
    for name in ("id_ed25519", "id_rsa"):
        p = Path.home() / ".ssh" / name
        if p.exists():
            return p
    return None


# --- SSH -------------------------------------------------------------------

def _connect(cfg: Config) -> paramiko.SSHClient:
    """SSH to cluster. Two-hop if gateway set. Key > password > prompt."""
    policy = paramiko.WarningPolicy() if cfg.trust_unknown_hosts else paramiko.RejectPolicy()
    key_path = _ssh_key_path(cfg)
    password = _load_password()
    auth: dict = {"username": cfg.user, "timeout": 30}
    if key_path:
        print(f"  Using SSH key: {key_path}")
        auth["key_filename"] = str(key_path)
    if password:
        auth["password"] = password
    if not key_path and not password:
        auth["password"] = getpass.getpass(f"Password for {cfg.user}@{cfg.host}: ")

    def _make_client(host, policy, sock=None):
        c = paramiko.SSHClient()
        c.load_system_host_keys()
        c.set_missing_host_key_policy(policy)
        kw = {**auth, **({"sock": sock} if sock else {})}
        c.connect(host, **kw)
        return c

    if cfg.gateway:
        print(f"  Connecting to {cfg.user}@{cfg.gateway} (gateway)...")
        gw = _make_client(cfg.gateway, policy)
        print(f"  Tunnelling to {cfg.host}...")
        chan = gw.get_transport().open_channel("direct-tcpip", (cfg.host, 22), ("127.0.0.1", 0))
        client = _make_client(cfg.host, policy, sock=chan)
        client._gateway = gw  # type: ignore[attr-defined]
        print("  Connected to login node.")
    else:
        print(f"  Connecting to {cfg.user}@{cfg.host}...")
        client = _make_client(cfg.host, policy)
        print("  Connected.")
    # Clear password from memory
    password = None  # noqa: F841
    auth.pop("password", None)
    return client


def _exec(client: paramiko.SSHClient, cmd: str, check: bool = True) -> str:
    """Execute command, return stdout, print stderr."""
    _, stdout, stderr = client.exec_command(cmd, timeout=600)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(f"  [stderr] {err.rstrip()}")
    if check and rc != 0:
        print(f"  Command failed (exit {rc}): {cmd}")
    return out


def _exec_live(client: paramiko.SSHClient, cmd: str) -> int:
    """Execute with live-streamed output. Returns exit code."""
    ch = client.get_transport().open_session()
    ch.exec_command(cmd)
    while True:
        if ch.recv_ready():
            print(ch.recv(4096).decode("utf-8", errors="replace"), end="", flush=True)
        if ch.recv_stderr_ready():
            print(f"[stderr] {ch.recv_stderr(4096).decode('utf-8', errors='replace')}",
                  end="", flush=True)
        if ch.exit_status_ready():
            while ch.recv_ready():
                print(ch.recv(4096).decode("utf-8", errors="replace"), end="", flush=True)
            break
        time.sleep(0.1)
    rc = ch.recv_exit_status()
    ch.close()
    return rc


# --- SFTP helpers ----------------------------------------------------------

def _sftp_makedirs(sftp: paramiko.SFTPClient, remote_dir: str):
    current = ""
    for part in PurePosixPath(remote_dir).parts:
        current = current + "/" + part if current else part
        if not current.startswith("/"):
            current = "/" + current
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def _sftp_upload_dir(sftp: paramiko.SFTPClient, local_dir: Path,
                     remote_dir: str, pattern: str = "*",
                     recursive: bool = True) -> tuple[int, int]:
    _sftp_makedirs(sftp, remote_dir)
    files = list(local_dir.rglob(pattern) if recursive else local_dir.glob(pattern))
    files = [f for f in files if f.is_file()]
    uploaded = 0
    for lf in files:
        rp = f"{remote_dir}/{lf.relative_to(local_dir).as_posix()}"
        _sftp_makedirs(sftp, str(PurePosixPath(rp).parent))
        try:
            if sftp.stat(rp).st_size == lf.stat().st_size:
                continue
        except FileNotFoundError:
            pass
        sftp.put(str(lf), rp)
        uploaded += 1
    return uploaded, len(files)


def _sftp_upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str):
    _sftp_makedirs(sftp, str(PurePosixPath(remote).parent))
    sftp.put(str(local), remote)


def _sftp_download_dir(sftp: paramiko.SFTPClient, remote_dir: str,
                       local_dir: Path, ext: tuple[str, ...] = ()) -> tuple[int, int]:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = sftp.listdir(remote_dir)
    except FileNotFoundError:
        return 0, 0
    if ext:
        files = [f for f in files if any(f.endswith(e) for e in ext)]
    dl = 0
    for fn in files:
        lp = local_dir / fn
        if not lp.exists():
            sftp.get(f"{remote_dir}/{fn}", str(lp))
            dl += 1
    return dl, len(files)


# --- Cluster flag helper ---------------------------------------------------

def _cflag(cfg: Config) -> str:
    return f"--cluster={cfg.cluster}" if cfg.cluster else ""


# --- Commands --------------------------------------------------------------

def cmd_test(cfg: Config):
    print("=" * 60, "\n  Testing SLURM cluster connection\n" + "=" * 60)
    client = _connect(cfg)
    _exec(client, "echo '  Hostname:' $(hostname)")
    _exec(client, "echo '  User:    ' $(whoami)")
    _exec(client, f"ls -d {cfg.remote_base} 2>/dev/null "
          f"&& echo '  Remote base: EXISTS' "
          f"|| echo '  Remote base: NOT FOUND (created on upload)'", check=False)
    print("\n  SLURM partitions:")
    _exec(client, f"sinfo {_cflag(cfg)} --summarize 2>/dev/null | head -20", check=False)
    client.close()
    print("\n  Connection test PASSED")


def cmd_setup_keys(cfg: Config):
    print("=" * 60, "\n  Setting up SSH keys\n" + "=" * 60)
    pub = Path.home() / ".ssh" / "id_ed25519.pub"
    if not pub.exists():
        pub = Path.home() / ".ssh" / "id_rsa.pub"
    if not pub.exists():
        print("  ERROR: No public key at ~/.ssh/id_ed25519.pub or id_rsa.pub")
        print("  Generate: ssh-keygen -t ed25519")
        return
    key_text = pub.read_text().strip()
    print(f"  Key: {pub}")
    client = _connect(cfg)
    _exec(client, "mkdir -p ~/.ssh && chmod 700 ~/.ssh")
    out = _exec(client, f"grep -c '{key_text[:40]}' ~/.ssh/authorized_keys 2>/dev/null || echo 0",
                check=False)
    if out.strip() != "0":
        print("  Key already registered.")
    else:
        _exec(client, f"echo '{key_text}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys")
        print("  Key copied. Key-based auth should now work.")
    client.close()


def cmd_upload(cfg: Config):
    print("=" * 60)
    print(f"  Uploading to cluster\n  Local:  {PROJECT_ROOT}\n  Remote: {cfg.remote_base}")
    print("=" * 60)
    client = _connect(cfg)
    sftp = client.open_sftp()

    print("\n[1/4] Creating remote directories...")
    for d in ("src", "data", "results", "logs"):
        _sftp_makedirs(sftp, f"{cfg.remote_base}/{d}")

    print("\n[2/4] Uploading source code...")
    for rd in UPLOAD_DIRS:
        ld = PROJECT_ROOT / rd
        if not ld.exists():
            print(f"  SKIP: {rd}/"); continue
        up, tot = _sftp_upload_dir(sftp, ld, f"{cfg.remote_base}/src/{rd}")
        print(f"  {rd}/: {up} updated / {tot} total")

    print("\n[3/4] Uploading build files...")
    for rf in UPLOAD_FILES:
        lf = PROJECT_ROOT / rf
        if not lf.exists():
            print(f"  SKIP: {rf}"); continue
        _sftp_upload_file(sftp, lf, f"{cfg.remote_base}/src/{rf}")
        print(f"  {rf}")

    print("\n[4/4] Uploading test data...")
    for rd in UPLOAD_DATA:
        ld = PROJECT_ROOT / rd
        if not ld.exists():
            print(f"  SKIP: {rd}/"); continue
        up, tot = _sftp_upload_dir(sftp, ld, f"{cfg.remote_base}/data/{Path(rd).name}")
        print(f"  {rd}: {up} updated / {tot} total")

    _exec(client, f"find {cfg.remote_base}/src -name '*.sh' -o -name '*.slurm' | "
          f"xargs sed -i 's/\\r$//' 2>/dev/null || true", check=False)
    print("\n  Verifying...")
    _exec(client, f"echo '  Source files:' $(find {cfg.remote_base}/src -type f | wc -l)", check=False)
    _exec(client, f"echo '  Data files:  ' $(find {cfg.remote_base}/data -type f 2>/dev/null | wc -l)",
          check=False)
    sftp.close(); client.close()
    print("\n" + "=" * 60 + "\n  Upload complete\n" + "=" * 60)
    print(f"\n  Next: uv run scripts/slurm/slurm_remote.py build")


def cmd_build(cfg: Config, profile: str = "arc"):
    print("=" * 60, f"\n  Building (profile: {profile})\n" + "=" * 60)
    client = _connect(cfg)
    out = _exec(client, f"test -f {cfg.remote_base}/src/CMakeLists.txt && echo OK || echo MISSING",
                check=False)
    if "MISSING" in out:
        print("  ERROR: Source not found. Run 'upload' first."); client.close(); return

    cl = f"#SBATCH --cluster={cfg.cluster}" if cfg.cluster else ""
    el = (f"#SBATCH --mail-user={cfg.email}\n#SBATCH --mail-type=END,FAIL") if cfg.email else ""
    rb = cfg.remote_base
    script = (f"#!/bin/bash\n#SBATCH --partition=interactive\n#SBATCH --time=01:00:00\n"
              f"#SBATCH --cpus-per-task=8\n#SBATCH --mem-per-cpu=4G\n"
              f"#SBATCH --job-name=dtwc-build\n#SBATCH --output={rb}/logs/build_%j.out\n"
              f"#SBATCH --error={rb}/logs/build_%j.err\n{cl}\n{el}\n\n"
              f"module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0 2>/dev/null || true\n\n"
              f"cd {rb}/src\nsource scripts/slurm/build-arc.sh {profile}\n")

    sp = f"{rb}/logs/build_job.sh"
    sftp = client.open_sftp()
    _sftp_makedirs(sftp, f"{rb}/logs")
    with sftp.open(sp, "w") as f:
        f.write(script)
    sftp.close()

    out = _exec(client, f"sbatch --parsable {sp}", check=True)
    jid = out.strip().split(";")[0]
    if not jid.isdigit():
        print(f"  ERROR: Cannot parse job ID: {out.strip()}"); client.close(); return
    print(f"\n  Build job: {jid}\n  Polling...")

    cf = _cflag(cfg)
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    while True:
        time.sleep(10)
        out = _exec(client, f"sacct -j {jid} {cf} --format=State --noheader 2>/dev/null | head -1",
                    check=False)
        state = out.strip().split()[0] if out.strip() else "UNKNOWN"
        if state in terminal:
            break
        print(f"  ... {state}", end="\r")

    print(f"\n  Build finished: {state}")
    if state == "COMPLETED":
        _exec(client, f"tail -5 {rb}/logs/build_{jid}.out 2>/dev/null", check=False)
    else:
        print("  Build log (last 20 lines):")
        _exec(client, f"tail -20 {rb}/logs/build_{jid}.out 2>/dev/null", check=False)
        print("  Error log:")
        _exec(client, f"tail -20 {rb}/logs/build_{jid}.err 2>/dev/null", check=False)
    client.close()


def _submit_job(cfg: Config, slurm_file: str, label: str, binary_check: str = ""):
    """Submit a .slurm job with preflight checks."""
    print("=" * 60, f"\n  Submitting: {label}\n" + "=" * 60)
    client = _connect(cfg)
    rscript = f"{cfg.remote_base}/src/{slurm_file}"
    out = _exec(client, f"test -f {rscript} && echo OK || echo MISSING", check=False)
    if "MISSING" in out:
        print(f"  ERROR: {slurm_file} not found. Run 'upload' first."); client.close(); return
    if binary_check:
        out = _exec(client, f"ls {cfg.remote_base}/src/{binary_check} 2>/dev/null | head -1",
                    check=False)
        if not out.strip():
            print(f"  ERROR: Binary not found ({binary_check}). Run 'build' first.")
            client.close(); return
        print(f"  Binary: {out.strip()}")
    parts = ["sbatch", "--parsable"]
    if cfg.cluster:
        parts.append(f"--cluster={cfg.cluster}")
    if cfg.email:
        parts.extend([f"--mail-user={cfg.email}", "--mail-type=END,FAIL"])
    parts.append(rscript)
    out = _exec(client, " ".join(parts), check=True)
    print(f"\n  Job submitted: {out.strip().split(';')[0]}")
    print(f"  Monitor: uv run scripts/slurm/slurm_remote.py status")
    client.close()


def cmd_submit_cpu(cfg: Config):
    _submit_job(cfg, "scripts/slurm/jobs/cpu_test.slurm", "CPU test",
                binary_check="build-*/bin/dtwc_cl")

def cmd_submit_gpu(cfg: Config):
    _submit_job(cfg, "scripts/slurm/jobs/gpu_test.slurm", "GPU test",
                binary_check="build-*/bin/dtwc_cl")

def cmd_submit_checkpoint(cfg: Config):
    _submit_job(cfg, "scripts/slurm/jobs/checkpoint_test.slurm", "Checkpoint test",
                binary_check="build-*/bin/dtwc_cl")

def cmd_submit_parquet(cfg: Config):
    _submit_job(cfg, "scripts/slurm/jobs/parquet_test.slurm", "Parquet test",
                binary_check="build-*/bin/dtwc_cl")


def cmd_status(cfg: Config):
    print("=" * 60, "\n  SLURM Job Status\n" + "=" * 60)
    client = _connect(cfg)
    cf = _cflag(cfg)
    print("\n  Queue:")
    _exec(client, f"squeue -u {cfg.user} {cf} 2>/dev/null | head -30", check=False)
    print("\n  Summary:")
    _exec(client,
          f"squeue -u {cfg.user} {cf} --noheader 2>/dev/null | "
          f"awk '{{st[$5]++}} END {{for(s in st) printf \"    %s: %d\\n\", s, st[s]}}'; "
          f"echo \"    Total: $(squeue -u {cfg.user} {cf} --noheader 2>/dev/null | wc -l)\"",
          check=False)
    client.close()


def cmd_stats(cfg: Config, job_id: str = ""):
    print("=" * 60, "\n  SLURM Job Statistics\n" + "=" * 60)
    client = _connect(cfg)
    cf = _cflag(cfg)
    if not job_id:
        print("\n  Recent jobs (last 24h):")
        _exec(client,
              f"sacct -u {cfg.user} {cf} "
              f"--format=JobID%-20,JobName%-20,State%-12,Elapsed%-12,MaxRSS%-12,ExitCode "
              f"--starttime=$(date -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || "
              f"date -v-24H '+%Y-%m-%dT%H:%M:%S' 2>/dev/null) 2>/dev/null | head -30",
              check=False)
    else:
        print(f"\n  Job ID: {job_id}")
        print("\n  Job summary:")
        _exec(client,
              f"sacct -j {job_id} {cf} "
              f"--format=JobID%-20,State%-12,Elapsed%-12,Start%-20,End%-20,MaxRSS%-12,ExitCode "
              f"--noheader 2>/dev/null | head -10", check=False)
        print("\n  Timing distribution:")
        _exec(client,
              f"sacct -j {job_id} {cf} --format=Elapsed --noheader 2>/dev/null | "
              f"grep -v '\\.' | sort | uniq -c | sort -rn | head -10", check=False)
        print("\n  Aggregate:")
        _exec(client,
              f"sacct -j {job_id} {cf} --format=State --noheader 2>/dev/null | "
              f"grep -v '\\.' | sort | uniq -c | sort -rn", check=False)
    client.close()


def cmd_download(cfg: Config):
    print("=" * 60, "\n  Downloading results\n" + "=" * 60)
    client = _connect(cfg)
    sftp = client.open_sftp()
    local_results = PROJECT_ROOT / "results" / "slurm"
    print("\n  Results...")
    dl, tot = _sftp_download_dir(sftp, f"{cfg.remote_base}/results", local_results)
    print(f"  Results: {dl} new / {tot} total")
    print("\n  Logs...")
    dl, tot = _sftp_download_dir(sftp, f"{cfg.remote_base}/logs",
                                 local_results / "logs", ext=(".out", ".err", ".log"))
    print(f"  Logs: {dl} new / {tot} total")
    sftp.close(); client.close()
    print(f"\n{'=' * 60}\n  Download complete\n  Local: {local_results}\n{'=' * 60}")


def cmd_ssh(cfg: Config, command: str):
    client = _connect(cfg)
    _exec(client, f"cd {cfg.remote_base} 2>/dev/null; {command}")
    client.close()


def cmd_interactive(cfg: Config):
    gw = f"ssh {cfg.user}@{cfg.gateway}\n     # Then: " if cfg.gateway else ""
    cf = f"--cluster={cfg.cluster} " if cfg.cluster else ""
    rb = cfg.remote_base
    print(f"""{'=' * 60}
  Interactive Session Guide
{'=' * 60}

  1. SSH to cluster:
     {gw}ssh {cfg.user}@{cfg.host}

  2. Request interactive session:
     srun {cf}--partition={cfg.partition} --time=02:00:00 \\
          --cpus-per-task=8 --mem-per-cpu=4G --pty bash

     # With GPU:
     srun {cf}--partition={cfg.partition} --time=02:00:00 \\
          --gres={cfg.gpu_gres} --cpus-per-task=8 --mem-per-cpu=4G --pty bash

  3. Build:
     module load CMake/3.27.6 GCC/13.2.0 CUDA/12.4.0
     cd {rb}/src
     source scripts/slurm/build-arc.sh arc

  4. Test:
     ctest --test-dir build-arc -C Release -j $(nproc)
{'=' * 60}""")


# --- CLI -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="SSH/SFTP remote helper for SLURM HPC clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Config: scripts/slurm/env.example")
    p.add_argument("--trust-unknown-hosts", action="store_true",
                   help="Accept unknown host keys (WarningPolicy)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("test", help="Test SSH connection")
    sub.add_parser("setup-keys", help="Copy SSH public key")
    sub.add_parser("upload", help="Upload source + test data")
    pb = sub.add_parser("build", help="Submit batch build job")
    pb.add_argument("--profile", default="arc",
                    choices=["arc", "htc-cpu", "htc-gpu", "htc-v4", "h100", "grace"])
    sub.add_parser("submit-cpu", help="Submit CPU test")
    sub.add_parser("submit-gpu", help="Submit GPU test")
    sub.add_parser("submit-checkpoint", help="Submit checkpoint test")
    sub.add_parser("submit-parquet", help="Submit Parquet test")
    sub.add_parser("status", help="SLURM queue")
    ps = sub.add_parser("stats", help="sacct statistics")
    ps.add_argument("--job", default="", help="Specific job ID")
    sub.add_parser("download", help="Download results + logs")
    px = sub.add_parser("ssh", help="Run command on cluster")
    px.add_argument("cmd", nargs="+", help="Command")
    sub.add_parser("interactive", help="Interactive session guide")

    args = p.parse_args()
    cfg = load_config(trust_unknown_hosts=args.trust_unknown_hosts)

    dispatch = {
        "test": lambda: cmd_test(cfg),
        "setup-keys": lambda: cmd_setup_keys(cfg),
        "upload": lambda: cmd_upload(cfg),
        "build": lambda: cmd_build(cfg, profile=args.profile),
        "submit-cpu": lambda: cmd_submit_cpu(cfg),
        "submit-gpu": lambda: cmd_submit_gpu(cfg),
        "submit-checkpoint": lambda: cmd_submit_checkpoint(cfg),
        "submit-parquet": lambda: cmd_submit_parquet(cfg),
        "status": lambda: cmd_status(cfg),
        "stats": lambda: cmd_stats(cfg, job_id=args.job),
        "download": lambda: cmd_download(cfg),
        "ssh": lambda: cmd_ssh(cfg, " ".join(args.cmd)),
        "interactive": lambda: cmd_interactive(cfg),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
